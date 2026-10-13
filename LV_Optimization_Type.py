#!/usr/bin/env python3

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import casadi as ca
import Atmosphere_Type as Atm_Type
import LV_Type_scaled as LV_Type
import LV_Plot as LVp
import pickle
from LaunchSites import parse_launch_site


class _IncompleteInitialGuess(ValueError):
    def __init__(self, phase, step):
        super().__init__(f'{phase}: initial trajectory became invalid at step {step}.')
        self.step = step


def _rollout_initial_guess(model, initial_state, steps, time_step, control, phase,
                          minimum_mass=0., stop_below_ground=False):
    """Return complete scaled states/controls, or reject the whole candidate."""
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError(f'{phase}: initial time step must be positive and finite.')
    states = np.empty((model.nx, steps + 1))
    controls = np.empty((model.nu, steps))
    states[:, 0] = np.asarray(initial_state, dtype=float).reshape(-1)
    for k in range(steps):
        physical = model.unscale_x(states[:, k]).full().flatten()
        if not np.isfinite(physical).all() or physical[-1] <= 0:
            raise _IncompleteInitialGuess(phase, k)
        controls[:, k] = np.asarray(control(physical), dtype=float).reshape(-1)
        if not np.isfinite(controls[:, k]).all():
            raise _IncompleteInitialGuess(phase, k)
        states[:, k+1] = model.dynamics_kp1(states[:, k], controls[:, k], time_step).full().flatten()
        following = model.unscale_x(states[:, k+1]).full().flatten()
        if not np.isfinite(following).all() or following[-1] <= 0:
            raise _IncompleteInitialGuess(phase, k+1)
        if k+1 < steps and (following[-1] < minimum_mass or
                           stop_below_ground and float(model.local_to_alt(following)) < -1.):
            raise _IncompleteInitialGuess(phase, k+1)
    return states, controls


class InitialGuessConstraintDiagnostics:
    """Label actual Opti constraints and evaluate them at the current initial values.

    Feasibility uses a 1e-6 tolerance in solver units. Display scales convert
    residuals to physical units without changing the optimization problem.
    """
    def __init__(self, opti, enabled=True, tolerance=1e-6):
        self.opti = opti
        self.enabled = enabled
        self.tolerance = tolerance
        self.constraints = {}
        self.next_index = opti.ng

    def add(self, phase, name, constraint, scale=1., unit='dimensionless', components=None, squared=False):
        self.opti.subject_to(constraint)
        # These constraints are scalar/vector relations. Counting their rows
        # avoids rebuilding Opti's full constraint graph after every addition.
        count = constraint.numel()
        start = self.next_index
        self.next_index += count
        scales = np.broadcast_to(np.asarray(scale, dtype=float), (count,))
        units = [unit] * count if isinstance(unit, str) else unit
        for offset in range(count):
            label = name if components is None else f'{name} [{components[offset]}]'
            self.constraints.setdefault(phase, []).append(
                (start + offset, label, float(scales[offset]), units[offset], squared))

    def report(self, phase, source='generated guess'):
        if not self.enabled:
            return None
        entries = self.constraints[phase]
        indices = [entry[0] for entry in entries]
        initial = self.opti.initial()
        try:
            values = np.asarray(self.opti.debug.value(self.opti.g[indices], initial)).reshape(-1)
            lower = np.asarray(self.opti.debug.value(self.opti.lbg[indices], initial)).reshape(-1)
            upper = np.asarray(self.opti.debug.value(self.opti.ubg[indices], initial)).reshape(-1)
        except RuntimeError as exc:
            print(f'\nInitial guess constraints — {phase} ({source}): could not evaluate: {exc}', flush=True)
            return {'phase': phase, 'source': source, 'evaluation_error': str(exc)}
        violations = []
        for entry, value, lb, ub in zip(entries, values, lower, upper):
            index, name, scale, unit, squared = entry
            invalid = not np.isfinite(value) or np.isnan(lb) or np.isnan(ub)
            residual = float(max(lb - value, value - ub, 0.)) if not invalid else float('nan')
            if not invalid and residual <= self.tolerance:
                continue
            def display(number):
                if squared and np.isfinite(number):
                    number = np.sqrt(number) if number >= 0 else float('nan')
                return float(number * scale)
            shown_value, shown_lb, shown_ub = display(value), display(lb), display(ub)
            amount = (float(max(shown_lb - shown_value, shown_value - shown_ub, 0.))
                      if not invalid else None)
            violations.append({'index': index, 'constraint': name, 'value': shown_value,
                               'lower': shown_lb, 'upper': shown_ub, 'violation': amount,
                               'unit': unit, 'scaled_residual': residual, 'nonfinite': invalid})
        print(f'\nInitial guess constraints — {phase} ({source}): '
              f'{len(violations)} / {len(entries)} violated or nonfinite '
              f'(solver tolerance {self.tolerance:g}).', flush=True)
        for item in violations:
            amount = 'NONFINITE — violation cannot be quantified' if item['nonfinite'] else (
                f"violation={item['violation']:.6g} {item['unit']}; "
                f"solver residual={item['scaled_residual']:.6g}")
            print(f"  g[{item['index']}] {item['constraint']}: "
                  f"value={item['value']:.6g}, bounds=[{item['lower']:.6g}, {item['upper']:.6g}] "
                  f"{item['unit']}; {amount}", flush=True)
        if not violations:
            print('  No constraint violations.', flush=True)
        return {'phase': phase, 'source': source, 'checked': len(entries), 'violations': violations}


@dataclass
class PythonMsg:
    '''
    base class for creating types and messages in python
    '''
    def __setattr__(self,key,value):
        '''
        Overloads default atribute-setting functionality
          to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally
          adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self,key):
            raise TypeError (f'Not allowed to add new field "{key}" to class {self}')
        else:
            object.__setattr__(self,key,value)

@dataclass
class VLType(PythonMsg):
    '''
    base class for creating types and messages in python
    '''
    N1: int = field(default = 60)
    N2: int = field(default = 10)
    N3: int = field(default = 60)
    N4: int = field(default = 15)
    init_time: float = field(default = 5.0)
    Target_Orbit: dict = field(default = None)
    RecoveryStrategy: str = field(default = 'None')

    booster: LV_Type.BoosterLaunchVehicle_2D = field(default = None)
    boostback: LV_Type.BoostBackBurn_2D = field(default = None)
    eci: LV_Type.LaunchVehicle_ECI = field(default = None)
    rocket_return: LV_Type.BoosterReturn_2D = field(default = None)
    atmosphere: Atm_Type = field(default = None)

    InitialGuess: dict = field(default = None)

    Solution: dict = field(default = None)

    ScenarioDispersion: LV_Type.DispesrionFactorsType = field(default = None)
    LV_Configuration: int = field(default=1)
    launch_site: dict = field(default=None)

@dataclass
class LV_Optimization(VLType):
    def __init__(self, ScenarioDispersion: LV_Type.DispesrionFactorsType, 
                         LV_Configuration: int = 1, launch_site=None):
        super().__init__()

        self.ScenarioDispersion = ScenarioDispersion
        self.LV_Configuration = LV_Configuration
        self.launch_site = parse_launch_site(launch_site)

        # Initialize Atmosphere:
        self.atmosphere = Atm_Type.AtmosphereType()

        # Initialize the rocket models
        self.booster = LV_Type.BoosterLaunchVehicle_2D(self.atmosphere, LV_Configuration, ScenarioDispersion, launch_site=self.launch_site)
        self.boostback = LV_Type.BoostBackBurn_2D(ScenarioDispersion, LV_Configuration, launch_site=self.launch_site)
        self.eci = LV_Type.LaunchVehicle_ECI(ScenarioDispersion, LV_Configuration, launch_site=self.launch_site)
        self.rocket_return = LV_Type.BoosterReturn_2D(self.atmosphere, LV_Configuration, ScenarioDispersion, launch_site=self.launch_site)

    def SolveOptimiztion(self, RecoveryStrategy: list = 'EXP',
                         payload_mass_predefined: float = -1.0, 
                         Target_Orbit = dict, 
                         Plot_interm: bool = False, 
                         print_ipopt: bool = False,
                         init_guess = None,
                         ipopt_options = None,
                         output_dir = 'Results',
                         initial_guess_diagnostics: bool = True):
        if RecoveryStrategy not in {'EXP', 'ASDS', 'RTLS'}:
            raise ValueError(f'Unknown recovery strategy: {RecoveryStrategy}')
        if init_guess is None:
            init_guess = 0.5
        elif isinstance(init_guess, dict):
            init_guess = init_guess.copy()
            if RecoveryStrategy != 'EXP' and init_guess.get('solution_units') != 'SI':
                # Legacy result files stored these two fields in solver units.
                init_guess['dt4_landing'] = init_guess['dt4_landing'] * self.rocket_return.scaleT
                init_guess['u4_reentry'] = float(np.asarray(init_guess['u4_reentry']).item()) * self.rocket_return.scaleU[0]

        Target_Orbit = dict(Target_Orbit)
        
        Target_Orbit["a"] = 0.5 * (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # semi-major axis
        Target_Orbit["e"] = (Target_Orbit["apogee"] - Target_Orbit["perigee"]) / (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # eccentricity    


        if RecoveryStrategy == 'EXP':
            LV_plot = LVp.LV_plot(self.booster, self.eci, None, None)
        elif RecoveryStrategy == 'ASDS':
            LV_plot = LVp.LV_plot(self.booster, self.eci, None, self.rocket_return)
        elif RecoveryStrategy == 'RTLS':
            LV_plot = LVp.LV_plot(self.booster, self.eci, self.boostback, self.rocket_return)

        # define the optimization problem
        opti = ca.Opti()
        diagnostics = InitialGuessConstraintDiagnostics(opti, enabled=initial_guess_diagnostics)
        local_components = ('x', 'z', 'vx', 'vz', 'mass')
        local_units = ('m', 'm', 'm/s', 'm/s', 'kg')
        eci_components = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'mass')
        eci_units = ('m', 'm', 'm', 'm/s', 'm/s', 'm/s', 'kg')
        phase1 = 'First-stage ascent'
        phase2 = 'Second stage before fairing separation'
        phase3 = 'Second stage after fairing separation'
        separation = 'Booster separation'
        boostback = 'Boostback burn'
        ballistic = 'Ballistic coast to reentry'
        reentry = 'Reentry burn'
        coast = 'Coast before landing'
        landing = 'Landing burn'

        # First Phase - Booster 
        if payload_mass_predefined > 0.0:
            payload_mass_scaled = payload_mass_predefined / self.booster.scaleX[4]
        else:
            payload_mass_scaled = opti.variable(1, 1)
            diagnostics.add(phase1, 'Payload mass lower bound', payload_mass_scaled >= 0.0,
                            scale=self.booster.scaleX[4], unit='kg')
        x1 = opti.variable(self.booster.nx, self.booster.N+1) # [x, z, vx, vz, m]
        u1 = opti.variable(self.booster.nu, self.booster.N) # [T_factor, alpha]
        dt1 = opti.variable(1)
        LaunchAz = opti.variable(1)

        payload_mass = payload_mass_scaled * self.booster.scaleX[4]

        # Set the initial state
        init_time = 5.0
        init_acc = self.booster.FirstStage_SL_Thrust / (self.booster.LV_total_mass + payload_mass) - self.booster.g0 * (self.booster.R0 / (self.booster.R0 + self.booster.LaunchAltitude))**2
        init_vz = init_acc * init_time
        init_vx = 0.0
        init_z = 0.5 * init_acc * init_time**2 + self.booster.LaunchAltitude
        init_x = 0.0
        init_m = self.booster.LV_total_mass + payload_mass - (self.booster.FirstStage_SL_Thrust / (self.booster.FirstStage_SL_Isp * self.booster.g0)) * init_time

        diagnostics.add(phase1, 'Initial state',
                        x1[:,0] == self.booster.scale_x(ca.vertcat(init_x, init_z, init_vx, init_vz, init_m)),
                        self.booster.scaleX, local_units, local_components)
        diagnostics.add(phase1, 'Time step lower bound', dt1 >= 0.1/self.booster.scaleT,
                        scale=self.booster.scaleT, unit='s')
        # set the constraints:
        for k in range(self.booster.N):
            # set the dynamics
            diagnostics.add(phase1, f'Dynamics k={k} → {k+1}',
                            x1[:,k+1] == self.booster.dynamics_kp1(x1[:,k], u1[:,k], dt1),
                            self.booster.scaleX, local_units, local_components)
            # set the thrust constraints
            diagnostics.add(phase1, f'Thrust factor upper bound k={k}', u1[0,k] <= 1.0)
            diagnostics.add(phase1, f'Thrust factor lower bound k={k}', u1[0,k] >= self.booster.FirstStage_MinThrust_Factor)
            diagnostics.add(phase1, f'Absolute angle of attack k={k}', u1[1,k]**2 <= 1.0,
                            scale=np.rad2deg(self.booster.FirstStage_MaxAlpha), unit='deg', squared=True)

            # set the state constraints
            uk = self.booster.unscale_u(u1[:,k])
            q = self.booster.dynamic_pressure_fun(x1[:,k])

            diagnostics.add(phase1, f'Dynamic pressure k={k}', q/self.booster.FirstStage_MaxDynamicPressure <= 1.0,
                            scale=self.booster.FirstStage_MaxDynamicPressure / 1000., unit='kPa')
            diagnostics.add(phase1, f'Absolute dynamic pressure × angle k={k}',
                            (uk[1] * q / self.booster.FirstStage_MaxQdynAlpha)**2 <= 1.0,
                            scale=self.booster.FirstStage_MaxQdynAlpha / 1000., unit='kPa·rad', squared=True)
            diagnostics.add(phase1, f'Payload acceleration k={k}',
                            self.booster.specific_acc_fun(x1[:,k], u1[:,k])/self.booster.Payload_Max_acc <= 1.0,
                            scale=self.booster.Payload_Max_acc, unit='m/s²')

        # set the final state constraints    
        x1f = self.booster.unscale_x(x1[:,self.booster.N])
        q1f = self.booster.dynamic_pressure_fun(x1[:,self.booster.N])
        diagnostics.add(phase1, 'Separation dynamic pressure',
                        q1f/self.booster.FirstStage_StageSeparationMaxDynamicPressure <= 1.0,
                        scale=self.booster.FirstStage_StageSeparationMaxDynamicPressure / 1000., unit='kPa')
        if RecoveryStrategy == 'EXP':
            diagnostics.add(phase1, 'Final mass lower bound',
                            x1f[4] >= self.eci.EmptyFirstStageMass + self.eci.SecondStage_FullMass + payload_mass, unit='kg')

        # Phase 2 - Second Stage, before fairing separation (if needed)
        x2 = opti.variable(self.eci.nx, self.eci.N[0]+1) # [x, y, z, vx, vy, vz, m]
        u2 = opti.variable(self.eci.nu, self.eci.N[0]) # [fx, fy, fz]
        dt2 = opti.variable(1)

        # Set the initial state - the final state of the boost phase transformed to ECI
        diagnostics.add(phase2, 'Time step lower bound', dt2 >= 0.1/self.eci.scaleT, self.eci.scaleT, 's')
        diagnostics.add(phase2, 'Launch azimuth lower bound', LaunchAz >= -np.pi/2, 180. / np.pi, 'deg')
        diagnostics.add(phase2, 'Launch azimuth upper bound', LaunchAz <= np.pi/2, 180. / np.pi, 'deg')
        eci_pos, eci_vel = self.eci.local_to_eci_func(ca.vertcat(x1f[0], 0.0, x1f[1]), ca.vertcat(x1f[2], 0.0, x1f[3]), LaunchAz)
        x2_init = ca.vertcat(eci_pos, eci_vel, self.eci.SecondStage_FullMass + payload_mass)
        x2_beforeCoast = self.eci.scale_x(x2_init)
        x2_afterCoast = self.eci.dynamics_kp1(x2_beforeCoast, 1e-8*np.ones((self.eci.nu,)), self.eci.SecondStage_CoastTimeAfterSep / self.eci.scaleT)
        
        diagnostics.add(phase2, 'Initial state after coast', x2[:,0] == x2_afterCoast,
                        self.eci.scaleX, eci_units, eci_components)
        for k in range(self.eci.N[0]):
            # set the dynamics
            diagnostics.add(phase2, f'Dynamics k={k} → {k+1}',
                            x2[:,k+1] == self.eci.dynamics_kp1(x2[:,k], u2[:,k], dt2),
                            self.eci.scaleX, eci_units, eci_components)
            # set the thrust constraints
            diagnostics.add(phase2, f'Thrust norm upper bound k={k}', ca.norm_2(u2[:,k]) <= 1.0)
            diagnostics.add(phase2, f'Thrust norm lower bound k={k}', ca.norm_2(u2[:,k]) >= self.eci.SecondStage_MinThrust_Factor)

        # set the final state constraints
        diagnostics.add(phase2, 'Fairing separation altitude',
                        self.eci.eci_to_alt(self.eci.unscale_x(x2[:,self.eci.N[0]]))/self.eci.FairingSeparationAltitude >= 1.0,
                        self.eci.FairingSeparationAltitude, 'm')

        ## phase 3 - Second Stage, after fairing separation
        x3 = opti.variable(self.eci.nx, self.eci.N[1]+1) # [x, y, z, vx, vy, vz, m]
        u3 = opti.variable(self.eci.nu, self.eci.N[1])
        dt3 = opti.variable(1)
        # Set the initial state - the final state of the boost phase transformed to ECI
        x3_init = ca.vertcat(x2[0:6,self.eci.N[0]], x2[6,self.eci.N[0]] - self.eci.FairingMass/ self.eci.scaleX[6])
        diagnostics.add(phase3, 'Time step lower bound', dt3 >= 0.1/self.eci.scaleT, self.eci.scaleT, 's')
        diagnostics.add(phase3, 'Initial state after fairing separation', x3[:,0] == x3_init,
                        self.eci.scaleX, eci_units, eci_components)
        for k in range(self.eci.N[1]):
            # set the dynamics
            diagnostics.add(phase3, f'Dynamics k={k} → {k+1}',
                            x3[:,k+1] == self.eci.dynamics_kp1(x3[:,k], u3[:,k], dt3),
                            self.eci.scaleX, eci_units, eci_components)
            # set the thrust constraints
            diagnostics.add(phase3, f'Thrust norm upper bound k={k}', ca.norm_2(u3[:,k]) <= 1.0)
            diagnostics.add(phase3, f'Thrust norm lower bound k={k}', ca.norm_2(u3[:,k]) >= self.eci.SecondStage_MinThrust_Factor)
            diagnostics.add(phase3, f'Payload acceleration k={k}',
                            self.eci.specific_acc_fun(x3[:,k], u3[:,k])/self.booster.Payload_Max_acc <= 1.0,
                            self.booster.Payload_Max_acc, 'm/s²')

        # set the final state constraints (Parking orbit)
        x3f = self.eci.unscale_x(x3[:,self.eci.N[1]])
        h = ca.cross(x3f[0:3], x3f[3:6])
        i = ca.acos(ca.fmin(ca.fmax(h[2] / ca.norm_2(h), -1.0), 1.0))
        e = ca.cross(x3f[3:6], h) / (self.eci.mu) - x3f[0:3] / ca.norm_2(x3f[0:3])
        eps = 0.5*ca.sumsqr(x3f[3:6]) - self.eci.mu / ca.norm_2(x3f[0:3])
        a = -self.eci.mu / (2.0*eps)
        apogee = a * (1.0 + ca.norm_2(e))
        perigee = a * (1.0 - ca.norm_2(e))

        # The optimzed orbit is a parking orbit where:
        # 1. The apogee is the same as the target orbit perigee
        # 2. The perigee is at least 100 km above the earth surface
        # 3. Need to calculate the propellent mass needed to reach the target orbit - raise the current perigee to the target orbit apogee
        # 3a. This manuever is done in the current apogee (which is the same as the target orbit perigee)

        v_apogee = ca.sqrt(self.eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / a))
        v_desired = ca.sqrt(self.eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["a"]))
        # propellent_mass_for_final_dv = x3f[6] * (1.0 - ca.exp(-ca.fabs(v_apogee - v_desired) / ((0.99*self.eci.SecondStage_Vac_Isp) * self.eci.g0)))
        
        # Current orbital-plane normal.
        n = h / ca.norm_2(h)
        cos_i_target = np.cos(Target_Orbit["i"])

        # At apogee, the position direction is opposite to e.
        # Only its squared vertical component is needed.
        e_squared = ca.sumsqr(e)
        rhat_z_squared = ca.if_else(e_squared > 1e-16, e[2]**2 / ca.fmax(e_squared, 1e-16), 0.0)  # Circular orbit: choose an equatorial crossing.

        # An attainable final plane must contain the burn position.
        rho_squared = ca.fmax(1.0 - rhat_z_squared, 0.0)
        target_margin = rho_squared - cos_i_target**2

        # Minimum plane rotation, allowing final RAAN to vary.
        normal_vertical = n[2]
        tangent_vertical_squared = ca.fmax(rho_squared - normal_vertical**2, 0.0)

        cos_plane_change = (normal_vertical * cos_i_target + ca.sqrt(tangent_vertical_squared * ca.fmax(target_margin, 0.0))) / ca.fmax(rho_squared, 1e-16)

        # At a geographic pole, every possible orbit is polar.
        cos_plane_change = ca.if_else(rho_squared > 1e-16, cos_plane_change, 1.0)
        cos_plane_change = ca.fmin(ca.fmax(cos_plane_change, -1.0), 1.0)

        delta_v_final = ca.sqrt((v_desired - v_apogee)**2 + 2.0 * v_apogee * v_desired * (1.0 - cos_plane_change))

        exhaust_velocity = (0.99 * self.eci.SecondStage_Vac_Isp * self.eci.g0)

        # If target inclination is geometrically unreachable at apogee,
        # zero remaining mass makes the existing mass constraint reject
        # this candidate. No additional subject_to constraint is added.
        mass_after_final_burn = ca.if_else(target_margin >= 0.0, x3f[6] * ca.exp(-delta_v_final / exhaust_velocity), 0.0)

        propellent_mass_for_final_dv = (x3f[6] - mass_after_final_burn)
       
        # v_final_apogee = ca.sqrt(self.rocket_eci.mu * (1.0 / Target_Orbit["a"]))
        # dv_for_inclination_fix = 2 * v_apogee * ca.sin(Target_Orbit["i"]/2.0)
        # propellent_mass_for_inclination_fix = (x3[6,self.rocket_eci.N[1]]-propellent_mass_for_final_dv) * (1.0 - ca.exp(-ca.fabs(dv_for_inclination_fix) / (self.rocket_eci.SecondStage_Vac_Isp * self.rocket_eci.g0)))


        # propellent_mass_for_final_dv = 0.0
        x3f_mass_scaled = (self.eci.SecondStage_EmptyMass + payload_mass + propellent_mass_for_final_dv) / self.eci.scaleX[6]
        diagnostics.add(phase3, 'Final mass including orbit-burn reserve',
                        x3[6,self.eci.N[1]] >= x3f_mass_scaled, self.eci.scaleX[6], 'kg')
        diagnostics.add(phase3, 'Parking apogee target', apogee/Target_Orbit["perigee"] == 1.0,
                        Target_Orbit['perigee'], 'm')
        diagnostics.add(phase3, 'Parking perigee lower bound',
                        perigee/(self.eci.R0 + self.eci.ParkingOrbit_PerigeeAlt) >= 1.0,
                        self.eci.R0 + self.eci.ParkingOrbit_PerigeeAlt, 'm')
        # opti.subject_to(h_final[2] / ca.norm_2(h_final) == np.cos(Target_Orbit["i"]))
        
        # if Target_Orbit["i"] == 0.0 or Target_Orbit["i"] == np.pi:
        #     # Equatorial planes require both horizontal angular-momentum
        #     # components to vanish, without dividing by zero or differentiating
        #     # acos at its singular endpoints. The sign selects pro/retrograde.
        #     h_scaled = h / np.sqrt(self.eci.mu * self.eci.R0)
        #     opti.subject_to(h_scaled[:2] == 0.0)
        #     opti.subject_to(h_scaled[2] >= 0.0 if Target_Orbit["i"] == 0.0 else h_scaled[2] <= 0.0)
        # else:
        #     opti.subject_to(i/Target_Orbit["i"] == 1.0)
            


    # Phase 4 - Booster Return:
        if RecoveryStrategy != 'EXP':
            reentry_altitude_min = 70000.
            reentry_thrust_max = 0.8
            landing_thrust_max = 1./3.
            landing_thrust_min = (0.75/3. * self.rocket_return.EmptyFirstStageMass
                                  * self.rocket_return.g0/self.rocket_return.scaleU[0])
            if self.rocket_return.FirstStage_MinThrust_Factor > reentry_thrust_max:
                raise ValueError('Reentry minimum thrust exceeds the permitted maximum of 0.8.')
            if landing_thrust_min > landing_thrust_max:
                raise ValueError('Landing minimum thrust exceeds the permitted maximum of 1/3.')
            if RecoveryStrategy == 'RTLS':
                dt4_boostback = opti.variable(1)
                x4_boostback = opti.variable(self.boostback.nx, 1) # [x, z, vx, vz, m]
                u4_boostback = opti.variable(self.boostback.nu, 1) # [Tx, Tz]
            x4_0 = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
            x4_before_reentry = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
            x4_after_reentryburn = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
            u4_reentry = opti.variable(self.rocket_return.nu, 1) # [T_factor]
            x4_before_landing = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
            x4_landing = opti.variable(self.rocket_return.nx, self.rocket_return.N+1) # [x, z, vx, vz, m]
            u4_landing = opti.variable(self.rocket_return.nu, self.rocket_return.N) # [T_factor]
            dt4_reentry = opti.variable(1)
            dt4_ballistic = opti.variable(1)
            dt4_before_landing = opti.variable(1)
            dt4_landing = opti.variable(1)

            # set the time step constraints
            if RecoveryStrategy == 'RTLS':
                diagnostics.add(boostback, 'Duration lower bound', dt4_boostback >= 1.0 / self.boostback.scaleT,
                                self.boostback.scaleT, 's')
            diagnostics.add(ballistic, 'Duration lower bound', dt4_ballistic >= 200.0 / self.rocket_return.scaleT,
                            self.rocket_return.scaleT, 's')
            diagnostics.add(reentry, 'Duration lower bound', dt4_reentry >= 0.1 / self.rocket_return.scaleT,
                            self.rocket_return.scaleT, 's')
            diagnostics.add(coast, 'Duration lower bound', dt4_before_landing >= 0.1 / self.rocket_return.scaleT,
                            self.rocket_return.scaleT, 's')
            diagnostics.add(landing, 'Time step lower bound', dt4_landing >= 0.1 / self.rocket_return.scaleT,
                            self.rocket_return.scaleT, 's')

            # Set the initial state
            x4_init_unscaled = ca.vertcat(x1f[0:4], x1f[4] - (self.eci.SecondStage_FullMass + payload_mass))
            
            # Boostback phase and ballistic phase
            if RecoveryStrategy == 'RTLS':
                diagnostics.add(separation, 'Initial state', x4_0 == self.boostback.scale_x(x4_init_unscaled),
                                self.boostback.scaleX, local_units, local_components)
                diagnostics.add(boostback, 'Dynamics',
                                x4_boostback == self.boostback.dynamics_kp1_M20(x4_0, u4_boostback, dt4_boostback),
                                self.boostback.scaleX, local_units, local_components)
                diagnostics.add(boostback, 'Thrust norm upper bound', ca.norm_2(u4_boostback) <= 1.0)

                diagnostics.add(ballistic, 'Dynamics',
                                x4_before_reentry == self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(self.boostback.unscale_x(x4_boostback)), 0, dt4_ballistic),
                                self.rocket_return.scaleX, local_units, local_components)
            else:
                diagnostics.add(separation, 'Initial state', x4_0 == self.rocket_return.scale_x(x4_init_unscaled),
                                self.rocket_return.scaleX, local_units, local_components)
                diagnostics.add(ballistic, 'Dynamics',
                                x4_before_reentry == self.rocket_return.dynamics_kp1_M50(x4_0, 0, dt4_ballistic),
                                self.rocket_return.scaleX, local_units, local_components)
            diagnostics.add(ballistic, 'Reentry start altitude lower bound',
                            self.rocket_return.local_to_alt(self.rocket_return.unscale_x(x4_before_reentry))/reentry_altitude_min >= 1.0,
                            reentry_altitude_min, 'm')

            # rentry burn
            diagnostics.add(reentry, 'Dynamics',
                            x4_after_reentryburn == self.rocket_return.dynamics_kp1_M20(x4_before_reentry, u4_reentry, dt4_reentry),
                            self.rocket_return.scaleX, local_units, local_components)
            ### set the thrust constraints
            diagnostics.add(reentry, 'Thrust factor upper bound', u4_reentry[0] <= reentry_thrust_max)
            diagnostics.add(reentry, 'Thrust factor lower bound', u4_reentry[0] >= self.rocket_return.FirstStage_MinThrust_Factor)

            ### After Reentry, before landing burn (ballistic)
            diagnostics.add(coast, 'Dynamics',
                            x4_before_landing == self.rocket_return.dynamics_kp1_M20(x4_after_reentryburn, 0, dt4_before_landing),
                            self.rocket_return.scaleX, local_units, local_components)

            N_before_landing_constraints = 20
            x4_before_landing_intren = x4_after_reentryburn
            for i in range(N_before_landing_constraints):
                x4_before_landing_intren = self.rocket_return.dynamics_kp1(x4_before_landing_intren, 0, dt4_before_landing/N_before_landing_constraints)
                diagnostics.add(coast, f'Dynamic pressure sample={i+1}/{N_before_landing_constraints}',
                                self.rocket_return.dynamic_pressure_fun(x4_before_landing_intren)/self.rocket_return.Booster_MaxDynamicPressure <= 1.0,
                                self.rocket_return.Booster_MaxDynamicPressure / 1000., 'kPa')
                diagnostics.add(coast, f'Heat flux sample={i+1}/{N_before_landing_constraints}',
                                self.rocket_return.heat_flux_fun(x4_before_landing_intren)/self.rocket_return.Booster_MaxHeatFlux <= 1.0,
                                self.rocket_return.Booster_MaxHeatFlux / 1000., 'kW/m²')

            ### Landing Burn:
            diagnostics.add(landing, 'Initial state', x4_landing[:,0] == x4_before_landing,
                            self.rocket_return.scaleX, local_units, local_components)
            for k in range(self.rocket_return.N):
                diagnostics.add(landing, f'Dynamics k={k} → {k+1}',
                                x4_landing[:,k+1] == self.rocket_return.dynamics_kp1(x4_landing[:,k], u4_landing[:,k], dt4_landing),
                                self.rocket_return.scaleX, local_units, local_components)
                diagnostics.add(landing, f'Thrust factor upper bound k={k}', u4_landing[0,k] <= landing_thrust_max)
                diagnostics.add(landing, f'Thrust factor lower bound k={k}',
                                u4_landing[0,k] >= landing_thrust_min)
                diagnostics.add(landing, f'Dynamic pressure k={k+1}',
                                self.rocket_return.dynamic_pressure_fun(x4_landing[:,k+1])/self.rocket_return.Booster_MaxDynamicPressure <= 1.0,
                                self.rocket_return.Booster_MaxDynamicPressure / 1000., 'kPa')
                diagnostics.add(landing, f'Heat flux k={k+1}',
                                self.rocket_return.heat_flux_fun(x4_landing[:,k+1])/self.rocket_return.Booster_MaxHeatFlux <= 1.0,
                                self.rocket_return.Booster_MaxHeatFlux / 1000., 'kW/m²')

            # final state constraints
            x4f_landing_unscaled = self.rocket_return.unscale_x(x4_landing[:,self.rocket_return.N])
            if RecoveryStrategy == 'RTLS':
                # opti.subject_to(ca.norm_2(x4_landing[0:2,self.rocket_return.N]) == 0.0)
                diagnostics.add(landing, 'Final downrange target', x4_landing[0,self.rocket_return.N] == 0.0,
                                self.rocket_return.scaleX[0], 'm')
                diagnostics.add(landing, 'Final local height target',
                                x4_landing[1,self.rocket_return.N] == self.rocket_return.LaunchAltitude/self.rocket_return.scaleX[1],
                                self.rocket_return.scaleX[1], 'm')
            else:
                # zero altitude at the end of the flight
                diagnostics.add(landing, 'Final altitude target', self.rocket_return.local_to_alt(x4f_landing_unscaled) == 0.0, unit='m')
            # minimal velocity at the end of the flight
            diagnostics.add(landing, 'Final speed upper bound',
                            ca.norm_2(x4_landing[2:4,self.rocket_return.N]) <= 10/self.rocket_return.scaleX[2],
                            self.rocket_return.scaleX[2], 'm/s')
            diagnostics.add(landing, 'Final mass lower bound',
                            x4_landing[4,self.rocket_return.N] >= self.rocket_return.EmptyFirstStageMass/self.rocket_return.scaleX[4],
                            self.rocket_return.scaleX[4], 'kg')

        # set the cost function
        gain = 5e2
        cost = 0.0
        cost += 0.5*gain*(ca.sumsqr(u1[0,1:]-u1[0,:-1]) + ca.sumsqr(u1[1,1:]-u1[1,:-1])) 
        cost += 0.5*gain*(ca.sumsqr(u2[0,1:]-u2[0,:-1])  + ca.sumsqr(u3[1,1:]-u3[1,:-1]))
        if RecoveryStrategy != 'EXP':
            cost += 0.05*gain*ca.sumsqr(u4_landing[1:]-u4_landing[:-1]) 
        cost += -(x3f[6] - propellent_mass_for_final_dv) # maximize the payload mass and propellant spares

        opti.minimize(cost)


        # set the solver
        
        hsl_library = os.environ.get('LV_IPOPT_HSL_LIBRARY', '/usr/local/lib/libcoinhsl.so')
        linear_solver = os.environ.get('LV_IPOPT_LINEAR_SOLVER', 'ma97' if Path(hsl_library).is_file() else 'mumps')
        ipopt_print_freq = 5 if print_ipopt is True else 0
        opts = {"print_time": 0,  # Print timing,
                "ipopt": {
                "linear_solver": linear_solver,
                "mu_strategy": "adaptive",  # "adaptive" or "adaptive" Strategy for updating the barrier parameter
                "tol": 1e-6,  # Convergence tolerance
                "max_iter": 400,  # Max iterations
                "print_level": ipopt_print_freq,  # Verbosity level
                'print_frequency_iter': 1,  # print_frequency_iter
                # "alpha_for_y": "min",  # Fraction-to-boundary rule parameter
                "timing_statistics": "no", # Enable timing statistics
                # "nlp_scaling_method": "none", # 'none' 'gradient-based', # Scaling method
                'nlp_scaling_max_gradient': 10,
                # "obj_scaling_factor": 1e-4, # Scaling factor for the objective function
                # "mu_init": 1e-11,  # 1e-1 Initial value of the barrier parameter
                # "mu_min": 1e-11,  # 1e-11 Smallest mu allowed — optimization stops when mu ≤ this
                # "mu_target": 0.0,  # 0.0 IPOPT drives mu towards this target; if 0, tries to reach exact KKT solution
                # "barrier_tol_factor": 5.0,  # Affects how mu and tolerances interact during convergence
                # "expect_infeasible_problem": "yes",  # More tolerant to infeasibility
                # "required_infeasibility_reduction": 1e-8,  # Less strict, delays restoration trigger
                # "soft_resto_pderror_reduction_factor": 0.1,  # Soft restoration less aggressive
                # "bound_relax_factor": 1e-8,  # Bound infeasibility relaxation very tight (prevents jumps)
                # "bound_push": 1e-8,  # Careful step near bounds
            }}
        if linear_solver.startswith('ma'):
            opts['ipopt']['hsllib'] = hsl_library
        opts['ipopt'].update(ipopt_options or {})
        if Plot_interm:
            LV_plot.plot_init()
            opti.callback(lambda i: LV_plot.plot_iteration(i, opti.debug.value(opti.debug.x), Target_Orbit, payload_mass_predefined))
        opti.solver('ipopt', opts)

        # calculate the initial guess for first stage 
        x1_guess = np.zeros((self.booster.nx, self.booster.N+1))
        u1_guess = np.zeros((self.booster.nu, self.booster.N))
        dt1_guess = (150.0-init_time) / self.booster.N / self.booster.scaleT
        LaunchAz_guess = np.pi/2.0 - Target_Orbit["i"]

        a_parking = 0.5 * (150.0*1e3 + self.eci.R0 + Target_Orbit["perigee"])
        v_apogee = ca.sqrt(self.eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / a_parking))
        v_desired = ca.sqrt(self.eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["a"]))
        v_parking = ca.sqrt(self.eci.mu * (2.0 / (150.0*1e3 + self.eci.R0) - 1.0 / a_parking))
        

        if RecoveryStrategy == 'RTLS':
            v3_0_guess = 2300.0
            v3_0_guess = 1500.0
        elif RecoveryStrategy == 'ASDS':
            v3_0_guess = 2000.0
            # v3_0_guess = 1750.0
        else:
            v3_0_guess = 2700.0

        if payload_mass_predefined <= 0.0:
            alpha = ca.exp(ca.fabs(v_parking-v3_0_guess) / (self.eci.SecondStage_Vac_Isp * self.eci.g0))
            if type(init_guess) == float:
                payload_mass_guess = init_guess*(self.eci.SecondStage_FullMass - alpha*self.eci.SecondStage_EmptyMass)/(alpha-1.0)
            else:
                payload_mass_guess = init_guess['payload_mass']
        else:
            payload_mass_guess = payload_mass_predefined
        m3_final = (self.eci.SecondStage_EmptyMass + payload_mass_guess)*(1.0 - ca.exp(-ca.fabs(v_apogee - v_desired) / (self.eci.SecondStage_Vac_Isp * self.eci.g0)))

        init_acc_guess = self.booster.FirstStage_SL_Thrust / (self.booster.LV_total_mass + payload_mass_guess) - self.booster.g0 * (self.booster.R0 / (self.booster.R0 + self.booster.LaunchAltitude))**2
        init_vz_guess = init_acc_guess * init_time
        init_vx_guess = 0.0
        init_z_guess = 0.5 * init_acc_guess * init_time**2 + self.booster.LaunchAltitude
        init_x_guess = 0.0
        init_m_guess = self.booster.LV_total_mass + payload_mass_guess - (self.booster.FirstStage_SL_Thrust / (self.booster.FirstStage_SL_Isp * self.booster.g0)) * init_time
        x0_guess = np.array([init_x_guess, init_z_guess, init_vx_guess, init_vz_guess, init_m_guess])

        if RecoveryStrategy == 'RTLS':
            dm = self.booster.FirstStagePropellentMass * 0.12
        elif RecoveryStrategy == 'ASDS':
            dm = self.booster.FirstStagePropellentMass * 0.035
        else:    
            dm = 0.0

        x1_guess[:,0] = self.booster.scale_x(x0_guess).full().flatten()
        iter=0
        while iter < 30:
            for k in range(self.booster.N):
                xk = self.booster.unscale_x(x1_guess[:,k]).full().flatten()
                alt = self.booster.local_to_alt(xk)

                if xk[1]<5000 or xk[1]>10000:
                    u1_guess[0,k] = 1.0
                else:
                    u1_guess[0,k] = 0.70
                if np.arctan2(xk[3], xk[2]) > 87.0*np.pi/180.0 and np.linalg.norm(xk[2:4]) > 75.0:
                    u1_guess[1,k] = -np.deg2rad(1.5) / self.booster.FirstStage_MaxAlpha
                elif alt > 15000:
                    u1_guess[1,k] = np.deg2rad(1.5) / self.booster.FirstStage_MaxAlpha
                else:
                    u1_guess[1,k] = 0


                max_u0 = self.booster.Payload_Max_acc*xk[4]/self.booster.FirstStage_Vac_Thrust
                u1_guess[0,k] = np.clip(u1_guess[0,k], self.booster.FirstStage_MinThrust_Factor, max_u0)
                
                x1_guess[:,k+1] = self.booster.dynamics_kp1(x1_guess[:,k], u1_guess[:,k], dt1_guess).full().flatten()
                acc = self.booster.specific_acc_fun(x1_guess[:,k], u1_guess[:,k])

            xN = self.booster.unscale_x(x1_guess[:,self.booster.N])
            alt_N1 = self.booster.local_to_alt(xN)
            Q = 0.5 * (xN[2]**2 + xN[3]**2) * self.atmosphere.rho_fun(alt_N1)

            # Keep the last fully propagated duration when the search is exhausted.
            if iter == 29:
                break
            if Q > self.booster.FirstStage_StageSeparationMaxDynamicPressure:
                dt1_guess += 0.5 / self.booster.N / self.booster.scaleT
                iter += 1
            elif xN[4] <= self.booster.FirstStage_EmptyMass + payload_mass_guess + dm:
                dt1_guess = max(0.1/self.booster.scaleT,
                                dt1_guess - 1.0 / self.booster.N / self.booster.scaleT)
                iter += 1
            else:
                break

        if type(payload_mass_scaled) == ca.MX:
            opti.set_initial(payload_mass_scaled, payload_mass_guess/self.booster.scaleX[4])
        opti.set_initial(dt1, dt1_guess)
        opti.set_initial(x1, x1_guess)
        opti.set_initial(u1, u1_guess)
        diagnostics.report(phase1)

        # calculate the initial guess for second stage before fairing separation, if needed
        dt2_guess = 20.0 / self.eci.N[0] / self.eci.scaleT
        x2_guess = np.zeros((self.eci.nx, self.eci.N[0]+1))
        u2_guess = np.zeros((self.eci.nu, self.eci.N[0]))
        x1f_guess = self.booster.unscale_x(x1_guess[:,self.booster.N])
        pos, vel = self.eci.local_to_eci_func(
            ca.vertcat(x1f_guess[0], 0.0, x1f_guess[1]),
            ca.vertcat(x1f_guess[2], 0.0, x1f_guess[3]), LaunchAz_guess)
        x2_init = ca.vertcat(pos, vel, self.eci.SecondStage_FullMass + payload_mass_guess)
        # Match the constraint: separate, transform to ECI, then coast until ignition.
        x2_guess[:,0] = self.eci.dynamics_kp1(
            self.eci.scale_x(x2_init), 1e-8*np.ones(self.eci.nu),
            self.eci.SecondStage_CoastTimeAfterSep / self.eci.scaleT).full().flatten()
        alt2_initial_guess = self.eci.eci_to_alt(self.eci.unscale_x(x2_guess[:,0]))
        iter=0
        while iter < 20:
            for k in range(self.eci.N[0]):
                xk = self.eci.unscale_x(x2_guess[:,k])
                u2_guess[0:3,k] = (x2_guess[3:6,k] / ca.norm_2(x2_guess[3:6,k])).full().flatten()
                x2_guess[:,k+1] = self.eci.dynamics_kp1(x2_guess[:,k], u2_guess[:,k], dt2_guess).full().flatten()
            xN = self.eci.unscale_x(x2_guess[:,self.eci.N[0]])
            alt = self.eci.eci_to_alt(xN)
            if alt >= self.eci.FairingSeparationAltitude or iter == 19:
                break
            else:
                dt2_guess += 4.0/self.eci.N[0]/self.eci.scaleT
                iter += 1

        opti.set_initial(LaunchAz, LaunchAz_guess)
        opti.set_initial(dt2, dt2_guess)
        opti.set_initial(x2, x2_guess)
        opti.set_initial(u2, u2_guess)
        diagnostics.report(phase2)

    # calculate the initial guess for second stage after fairing separation
        dt3_guess = 370.0/self.eci.N[1]/self.eci.scaleT
        x3_initial = x2_guess[:,-1].copy()
        x3_initial[6] -= self.eci.FairingMass / self.eci.scaleX[6]
        minimum_stage3_mass = float(m3_final + self.eci.SecondStage_EmptyMass + payload_mass_guess)
        last_complete_stage3 = None
        iter=0

        alpha3_init = 0.0
        v3_max = ca.sqrt(self.eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["apogee"]))
        while iter < 50:
            def stage3_control(xk):
                h = np.cross(xk[0:3], xk[3:6])
                h = h / np.linalg.norm(h)
                direction = xk[3:6] / np.linalg.norm(xk[3:6])
                alpha3 = -alpha3_init
                direction = (direction * np.cos(alpha3) + np.cross(h, direction) * np.sin(alpha3)
                             + h * np.dot(h, direction) * (1 - np.cos(alpha3)))
                acc = self.eci.SecondStage_Thrust / xk[6]
                throttle = np.clip(self.eci.Payload_Max_acc/acc,
                                   self.eci.SecondStage_MinThrust_Factor, 1.)
                return direction * throttle

            try:
                x3_guess, u3_guess = _rollout_initial_guess(
                    self.eci, x3_initial, self.eci.N[1], float(dt3_guess), stage3_control,
                    phase3, minimum_mass=minimum_stage3_mass)
            except _IncompleteInitialGuess as exc:
                dt3_guess = max(0.1/self.eci.scaleT,
                                dt3_guess * max(0.1, min(0.9, exc.step / self.eci.N[1])))
                iter += 1
                continue
            last_complete_stage3 = (x3_guess, u3_guess, float(dt3_guess))
            xkp1 = self.eci.unscale_x(x3_guess[:,-1]).full().flatten()
            h_guess = np.cross(xkp1[0:3], xkp1[3:6])
            e_guess = ca.cross(xkp1[3:6], h_guess) / self.eci.mu - xkp1[0:3] / ca.norm_2(xkp1[0:3])
            eps_guess = 0.5*ca.sumsqr(xkp1[3:6]) - self.eci.mu / ca.norm_2(xkp1[0:3])
            a_guess = -self.eci.mu / (2.0*eps_guess)
            perigee_guess = a_guess * (1.0 - ca.norm_2(e_guess))
            apogee_guess = a_guess * (1.0 + ca.norm_2(e_guess))
            if iter == 49:
                break
            if xkp1[6] < minimum_stage3_mass:
                dt3_guess = max(0.1/self.eci.scaleT, dt3_guess - 0.25/self.eci.scaleT)
                iter += 1
            elif apogee_guess < self.eci.ParkingOrbit_PerigeeAlt + self.eci.R0 or a_guess < 0.5*(Target_Orbit['perigee']+self.eci.R0+200.0*1e3):
                dt3_guess += 0.1/self.eci.scaleT
                iter += 1
            elif perigee_guess < self.eci.R0+10*10**3:
                alpha3_init += np.deg2rad(1.0)
                # dt3_guess += 0.
                iter += 1
            elif apogee_guess > Target_Orbit['perigee'] + 150.0*1e3:
                dt3_guess = max(0.1/self.eci.scaleT, dt3_guess - 0.1/self.eci.scaleT)
                iter += 1
            else:
                break
        if last_complete_stage3 is None:
            raise ValueError('Second-stage initial guess could not generate a complete finite trajectory.')
        x3_guess, u3_guess, dt3_guess = last_complete_stage3
        opti.set_initial(dt3, dt3_guess)
        opti.set_initial(x3, x3_guess)
        opti.set_initial(u3, u3_guess)
        diagnostics.report(phase3)

        # calculate the initial guess for self.booster return
        if RecoveryStrategy != 'EXP':
            # calculate the initial guess for self.booster return
            x4_0_guess = np.zeros((self.rocket_return.nx, 1))
            x1f_guess_unscaled = self.booster.unscale_x(x1_guess[:,self.booster.N]).full().flatten()
            x4_0_guess[0:4,0] = x1f_guess_unscaled[0:4]
            x4_0_guess[4,0] = x1f_guess_unscaled[4] - self.eci.SecondStage_FullMass - payload_mass_guess
            return_model = self.boostback if RecoveryStrategy == 'RTLS' else self.rocket_return
            opti.set_initial(x4_0, return_model.scale_x(x4_0_guess))
            diagnostics.report(separation)
            if RecoveryStrategy == 'RTLS':
                dt4_boostback_guess = 22.0 / self.boostback.scaleT
                x4_boostback_guess = np.zeros((self.boostback.nx, 1))
                u4_boostback_guess = np.zeros((self.boostback.nu, 1))
                iter = 0
                while iter < 50:
                    u4_boostback_guess[0] = -np.cos(np.deg2rad(-10.0))
                    u4_boostback_guess[1] = -np.sin(np.deg2rad(-10.0))
                    x4_boostback_guess = self.boostback.dynamics_kp1_M20(self.boostback.scale_x(x4_0_guess), u4_boostback_guess, dt4_boostback_guess).full().flatten()
                    x4_boostback_guess_unscaled = self.boostback.unscale_x(x4_boostback_guess)
                    alt = self.booster.local_to_alt(x4_boostback_guess_unscaled)
                    vel = np.linalg.norm(x4_boostback_guess_unscaled[2:4])
                    t_bal = (-x4_boostback_guess_unscaled[3]  + np.sqrt(x4_boostback_guess_unscaled[3]**2 +2*self.boostback.g0*alt))/(0.5*self.boostback.g0)
                    x_bal = x4_boostback_guess_unscaled[0] + x4_boostback_guess_unscaled[2]*400
                    if iter == 49:
                        break
                    if abs(x_bal) > 100.0:
                        dt4_boostback_guess = max(1./self.boostback.scaleT,
                            dt4_boostback_guess + np.clip(x_bal / max(abs(float(x4_boostback_guess_unscaled[2])), 1.)/400, -1.0, 1.0) / self.boostback.scaleT)
                        iter += 1
                    else:
                        break

                opti.set_initial(dt4_boostback, dt4_boostback_guess)
                opti.set_initial(x4_boostback, x4_boostback_guess)
                opti.set_initial(u4_boostback, u4_boostback_guess)
                diagnostics.report(boostback)

            # calculate the initial guess for self.booster return
            dt4_ballistic_guess = 300 / self.rocket_return.scaleT
            reentry_altitude_target = reentry_altitude_min + 1000.
            iter=0
            while iter < 50:
                if RecoveryStrategy == 'RTLS':
                    x4_before_reentry_guess = self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(x4_boostback_guess_unscaled), 0, dt4_ballistic_guess)
                else:
                    x4_before_reentry_guess = self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(x4_0_guess), 0, dt4_ballistic_guess)
                x4_before_reentry_guess_unscaled = self.rocket_return.unscale_x(x4_before_reentry_guess)
                alt = self.rocket_return.local_to_alt(x4_before_reentry_guess_unscaled)
                vel = ca.norm_2(x4_before_reentry_guess_unscaled[2:4,0])
                if iter == 49:
                    break
                if abs(alt - reentry_altitude_target) > 10:
                    dt4_ballistic_guess = max(200./self.rocket_return.scaleT,
                        dt4_ballistic_guess + np.clip((alt - reentry_altitude_target)/max(abs(float(x4_before_reentry_guess_unscaled[3,0])), 1.), -10.0, 10.0) / self.rocket_return.scaleT)
                    iter += 1
                else:
                    break
            
            opti.set_initial(dt4_ballistic, dt4_ballistic_guess)
            opti.set_initial(x4_before_reentry, x4_before_reentry_guess)
            diagnostics.report(ballistic)

            u4_reentry_guess = reentry_thrust_max
            dt4_reentry_guess = 1.0 / self.rocket_return.scaleT
            iter = 0
            v_final = 1300.0
            while iter < 50:
                x4_after_reentry_guess = self.rocket_return.dynamics_kp1_M20(x4_before_reentry_guess, u4_reentry_guess, dt4_reentry_guess)
                x4_after_reentry_guess_unscaled = self.rocket_return.unscale_x(x4_after_reentry_guess)
                alt = self.rocket_return.local_to_alt(x4_after_reentry_guess_unscaled)
                vel = ca.norm_2(x4_after_reentry_guess_unscaled[2:4,0])
                mass = x4_after_reentry_guess_unscaled[4,0]
                if iter == 49:
                    break
                if abs(vel- v_final) > 10:
                    dt4_reentry_guess = max(0.1/self.rocket_return.scaleT,
                        dt4_reentry_guess + 0.5 * np.sign(vel - v_final) / self.rocket_return.scaleT)
                else:
                    break
                iter += 1

            opti.set_initial(dt4_reentry, dt4_reentry_guess)
            opti.set_initial(x4_after_reentryburn, x4_after_reentry_guess)
            opti.set_initial(u4_reentry, u4_reentry_guess)
            diagnostics.report(reentry)

            # after reentry burn before landing burn
            iter = 0
            dt4_before_landing_guess = 50.0 / self.rocket_return.scaleT
            alt4_before_landing = 1000.0
            while iter < 50:
                x4_before_landing_guess = self.rocket_return.dynamics_kp1_M20(x4_after_reentry_guess, 0, dt4_before_landing_guess)
                x4_before_landing_guess_unscaled = self.rocket_return.unscale_x(x4_before_landing_guess).full().flatten()
                alt = self.rocket_return.local_to_alt(x4_before_landing_guess_unscaled)
                vel = np.linalg.norm(x4_before_landing_guess_unscaled[2:4])
                if iter == 49:
                    break
                if abs(alt - alt4_before_landing) > 10:
                    dt4_before_landing_guess = max(0.1/self.rocket_return.scaleT,
                        dt4_before_landing_guess + 0.9*(alt - alt4_before_landing)/max(vel, 1.) / self.rocket_return.scaleT)
                    iter += 1
                else:
                    break

            opti.set_initial(dt4_before_landing, dt4_before_landing_guess)
            opti.set_initial(x4_before_landing, x4_before_landing_guess)
            diagnostics.report(coast)

            # initial guess for the landing burn
            vel = np.linalg.norm(x4_before_landing_guess_unscaled[2:4])
            alt  = self.rocket_return.local_to_alt(x4_before_landing_guess_unscaled)
            landing_deceleration = (self.rocket_return.scaleU[0] * landing_thrust_max
                                    / x4_before_landing_guess_unscaled[4] - self.rocket_return.g0)
            dt4_landing_guess = max(0.1/self.rocket_return.scaleT,
                vel/max(landing_deceleration, 1.) / self.rocket_return.N / self.rocket_return.scaleT)
            iter = 0
            thrust_factor = 1.0
            last_complete_landing = None
            while iter < 20:
                def landing_control(xk):
                    speed = max(np.linalg.norm(xk[2:4]), 1.)
                    height = float(self.rocket_return.local_to_alt(xk))
                    safe_height = max(height, 1.)
                    g_vec = -np.array([xk[0], self.rocket_return.R0 + xk[1]])
                    g_vec /= np.linalg.norm(g_vec)
                    g = self.rocket_return.g0 * (self.rocket_return.R0/(self.rocket_return.R0 + xk[1]))**2
                    g_v = np.dot(g_vec, xk[2:4])/speed * g
                    thrust = g_v * xk[4]/self.rocket_return.scaleU[0]
                    if speed > 30.0 and height > 50.0:
                        thrust += (3*speed**2)/(2*safe_height)*xk[4]/self.rocket_return.scaleU[0]
                    if height < 50.0:
                        thrust += 0.5*(speed-0.5)**2/safe_height*xk[4]/self.rocket_return.scaleU[0]
                    return [np.clip(thrust_factor * thrust, landing_thrust_min, landing_thrust_max)]

                try:
                    x4_landing_guess, u4_landing_guess = _rollout_initial_guess(
                        self.rocket_return, x4_before_landing_guess, self.rocket_return.N,
                        float(dt4_landing_guess), landing_control, landing, stop_below_ground=True)
                except _IncompleteInitialGuess as exc:
                    dt4_landing_guess = max(0.1/self.rocket_return.scaleT,
                        dt4_landing_guess * max(0.1, min(0.9, exc.step / self.rocket_return.N)))
                    iter += 1
                    continue
                last_complete_landing = (x4_landing_guess, u4_landing_guess, float(dt4_landing_guess))
                xkp1 = self.rocket_return.unscale_x(x4_landing_guess[:,-1]).full().flatten()
                alt = self.rocket_return.local_to_alt(xkp1)
                vel = np.linalg.norm(xkp1[2:4])
                if iter == 19:
                    break
                if alt > 10:
                    dt4_landing_guess += alt/max(vel, 1.)/self.rocket_return.N / self.rocket_return.scaleT
                    # thrust_factor *= 1.01
                    iter += 1
                # elif vel > 20:
                #     thrust_factor += self.rocket_return.FirstStage_EmptyMass/self.rocket_return.FirstStage_SL_Thrust
                #     iter += 1
                elif alt < -1.0:
                    thrust_factor *= 1.1
                    iter += 1
                else:
                    break
            if last_complete_landing is None:
                raise ValueError('Landing initial guess could not generate a complete finite trajectory.')
            x4_landing_guess, u4_landing_guess, dt4_landing_guess = last_complete_landing
            opti.set_initial(dt4_landing, dt4_landing_guess)
            opti.set_initial(x4_landing, x4_landing_guess)
            opti.set_initial(u4_landing, u4_landing_guess)
            diagnostics.report(landing)

        if type(init_guess) == dict:
            if type(payload_mass_scaled) == ca.MX:
                opti.set_initial(payload_mass_scaled, init_guess['payload_mass']/self.booster.scaleX[4])
            opti.set_initial(dt1, init_guess['dt1']/self.booster.scaleT)
            opti.set_initial(dt2, init_guess['dt2']/self.eci.scaleT)
            opti.set_initial(dt3, init_guess['dt3']/self.eci.scaleT)
            opti.set_initial(LaunchAz, init_guess['LaunchAz'])
            for k in range(init_guess['x1'].shape[1]):
                opti.set_initial(x1[:,k], self.booster.scale_x(init_guess['x1'][:,k]))
            for k in range(init_guess['u1'].shape[1]):
                opti.set_initial(u1[:,k], self.booster.scale_u(init_guess['u1'][:,k]))
            for k in range(init_guess['x2'].shape[1]):
                opti.set_initial(x2[:,k], self.eci.scale_x(init_guess['x2'][:,k]))
            for k in range(init_guess['u2'].shape[1]):
                opti.set_initial(u2[:,k], self.eci.scale_u(init_guess['u2'][:,k]))
            for k in range(init_guess['x3'].shape[1]):
                opti.set_initial(x3[:,k], self.eci.scale_x(init_guess['x3'][:,k]))
            for k in range(init_guess['u3'].shape[1]):
                opti.set_initial(u3[:,k], self.eci.scale_u(init_guess['u3'][:,k]))
            if RecoveryStrategy != 'EXP':
                opti.set_initial(dt4_ballistic, init_guess['dt4_ballistic']/self.rocket_return.scaleT)
                opti.set_initial(dt4_reentry, init_guess['dt4_reentry']/self.rocket_return.scaleT)
                opti.set_initial(dt4_before_landing, init_guess['dt4_before_landing']/self.rocket_return.scaleT)
                opti.set_initial(dt4_landing, init_guess['dt4_landing']/self.rocket_return.scaleT)
                opti.set_initial(x4_before_reentry, self.rocket_return.scale_x(init_guess['x4_before_reentry']))
                opti.set_initial(x4_after_reentryburn, self.rocket_return.scale_x(init_guess['x4_after_reentryburn']))
                opti.set_initial(u4_reentry, self.rocket_return.scale_u(np.asarray(init_guess['u4_reentry']).reshape(-1)))
                opti.set_initial(x4_before_landing, self.rocket_return.scale_x(init_guess['x4_before_landing']))
                for k in range(init_guess['x4_landing'].shape[1]):
                    opti.set_initial(x4_landing[:,k], self.rocket_return.scale_x(init_guess['x4_landing'][:,k]))
                for k in range(init_guess['u4_landing'].shape[0]):
                    opti.set_initial(u4_landing[k], self.rocket_return.scale_u([init_guess['u4_landing'][k]]))
                if RecoveryStrategy == 'RTLS':
                    opti.set_initial(x4_0, self.boostback.scale_x(init_guess['x4_0']))
                    opti.set_initial(dt4_boostback, init_guess['dt4_boostback']/self.boostback.scaleT)
                    opti.set_initial(x4_boostback, self.boostback.scale_x(init_guess['x4_boostback']))
                    opti.set_initial(u4_boostback, self.boostback.scale_u(init_guess['u4_boostback']))
                else:
                    opti.set_initial(x4_0, self.rocket_return.scale_x(init_guess['x4_0']))
            for phase in diagnostics.constraints:
                diagnostics.report(phase, source='supplied warm start')
        dispersion_values = {key: float(value) for key, value in asdict(self.ScenarioDispersion).items()}
        case_values = {'configuration': self.LV_Configuration, 'dispersion': dispersion_values,
                       'launch_site': self.launch_site,
                       'apogee': float(Target_Orbit['apogee']), 'perigee': float(Target_Orbit['perigee']),
                       'inclination': float(Target_Orbit['i']), 'payload': float(payload_mass_predefined)}
        case_id = hashlib.sha256(json.dumps(case_values, sort_keys=True).encode()).hexdigest()[:16]
        configuration_tag = 'Custom' if isinstance(self.LV_Configuration, dict) else self.LV_Configuration
        filename = (f"LV_{RecoveryStrategy}_apogee_{0.001*(Target_Orbit['apogee']-self.booster.R0):.0f}_km_"
                    f"perigee_{0.001*(Target_Orbit['perigee']-self.booster.R0):.0f}km_"
                    f"inc_{np.rad2deg(Target_Orbit['i']):.1f}deg_"
                    f"PL_{max(0,payload_mass_predefined):.0f}kg_Config{configuration_tag}_{case_id}")
        solve_error = None
        try:
            sol = opti.solve()

            u1_scaled_sol = np.array(sol.value(u1))
            x1_scaled_sol = np.array(sol.value(x1))
            dt1_scaled_sol = np.array(sol.value(dt1))
            u2_scaled_sol = np.array(sol.value(u2))
            x2_scaled_sol = np.array(sol.value(x2))
            dt2_scaled_sol = np.array(sol.value(dt2))
            dt3_scaled_sol = np.array(sol.value(dt3))
            x3_scaled_sol = np.array(sol.value(x3))
            u3_scaled_sol = np.array(sol.value(u3))
            if type(payload_mass_scaled) == ca.MX:
                payload_mass_scaled_sol = np.array(sol.value(payload_mass_scaled))
            LaunchAz_sol = np.array(sol.value(LaunchAz))
            propellent_mass_for_final_dv_sol = np.array(sol.value(propellent_mass_for_final_dv))
            if RecoveryStrategy != 'EXP':
                dt4_reentry_scaled_sol = np.array(sol.value(dt4_reentry))
                dt4_ballistic_scaled_sol = np.array(sol.value(dt4_ballistic))
                dt4_before_landing_scaled_sol = np.array(sol.value(dt4_before_landing))
                dt4_landing_scaled_sol = np.array(sol.value(dt4_landing))
                x4_before_reentry_scaled_sol = np.array(sol.value(x4_before_reentry))
                x4_after_reentry_scaled_sol = np.array(sol.value(x4_after_reentryburn))
                u4_reentry_scaled_sol = np.array(sol.value(u4_reentry))
                x4_before_landing_scaled_sol = np.array(sol.value(x4_before_landing))
                x4_landing_scaled_sol = np.array(sol.value(x4_landing))
                u4_landing_scaled_sol = np.array(sol.value(u4_landing))
                x4_0_scaled_sol = np.array(sol.value(x4_0))
                if RecoveryStrategy == 'RTLS':
                    dt4_boostback_scaled_sol = np.array(sol.value(dt4_boostback))
                    x4_boostback_scaled_sol = np.array(sol.value(x4_boostback))
                    u4_boostback_scaled_sol = np.array(sol.value(u4_boostback))


        except RuntimeError as exc:
            solve_error = str(exc)
            filename += '_Failed'
            u1_scaled_sol = np.array(opti.debug.value(u1))
            x1_scaled_sol = np.array(opti.debug.value(x1))
            dt1_scaled_sol = np.array(opti.debug.value(dt1))
            u2_scaled_sol = np.array(opti.debug.value(u2))
            x2_scaled_sol = np.array(opti.debug.value(x2))
            dt2_scaled_sol = np.array(opti.debug.value(dt2))
            dt3_scaled_sol = np.array(opti.debug.value(dt3))
            x3_scaled_sol = np.array(opti.debug.value(x3))
            u3_scaled_sol = np.array(opti.debug.value(u3))
            if type(payload_mass_scaled) == ca.MX:
                payload_mass_scaled_sol = np.array(opti.debug.value(payload_mass_scaled))
            LaunchAz_sol = np.array(opti.debug.value(LaunchAz))
            propellent_mass_for_final_dv_sol = np.array(opti.debug.value(propellent_mass_for_final_dv))
            if RecoveryStrategy != 'EXP':
                dt4_reentry_scaled_sol = np.array(opti.debug.value(dt4_reentry))
                dt4_ballistic_scaled_sol = np.array(opti.debug.value(dt4_ballistic))
                dt4_before_landing_scaled_sol = np.array(opti.debug.value(dt4_before_landing))
                dt4_landing_scaled_sol = np.array(opti.debug.value(dt4_landing))
                x4_before_reentry_scaled_sol = np.array(opti.debug.value(x4_before_reentry))
                x4_after_reentry_scaled_sol = np.array(opti.debug.value(x4_after_reentryburn))
                u4_reentry_scaled_sol = np.array(opti.debug.value(u4_reentry))
                x4_before_landing_scaled_sol = np.array(opti.debug.value(x4_before_landing))
                x4_landing_scaled_sol = np.array(opti.debug.value(x4_landing))
                u4_landing_scaled_sol = np.array(opti.debug.value(u4_landing))
                x4_0_scaled_sol = np.array(opti.debug.value(x4_0))
                if RecoveryStrategy == 'RTLS':
                    dt4_boostback_scaled_sol = np.array(opti.debug.value(dt4_boostback))
                    x4_boostback_scaled_sol = np.array(opti.debug.value(x4_boostback))
                    u4_boostback_scaled_sol = np.array(opti.debug.value(u4_boostback))

        if Plot_interm:
            LV_plot.plot_close()

        if type(payload_mass_scaled) == ca.MX:
            payload_mass_sol = payload_mass_scaled_sol * self.booster.scaleX[4]
        else:
            payload_mass_scaled_sol = payload_mass
            payload_mass_sol = payload_mass_scaled_sol
        dt1_sol = dt1_scaled_sol * self.booster.scaleT
        dt2_sol = dt2_scaled_sol * self.eci.scaleT
        dt3_sol = dt3_scaled_sol * self.eci.scaleT
        u1_sol = np.zeros_like(u1_scaled_sol)
        for i in range(u1_scaled_sol.shape[1]):
            u1_sol[:,i] = self.booster.unscale_u(u1_scaled_sol[:,i]).full().flatten()
        x1_sol = np.zeros_like(x1_scaled_sol)
        for i in range(x1_scaled_sol.shape[1]):
            x1_sol[:,i] = self.booster.unscale_x(x1_scaled_sol[:,i]).full().flatten()
        u2_sol = np.zeros_like(u2_scaled_sol)
        for i in range(u2_scaled_sol.shape[1]):
            u2_sol[:,i] = self.eci.unscale_u(u2_scaled_sol[:,i]).full().flatten()
        x2_sol = np.zeros_like(x2_scaled_sol)
        for i in range(x2_scaled_sol.shape[1]):
            x2_sol[:,i] = self.eci.unscale_x(x2_scaled_sol[:,i]).full().flatten()
        x3_sol = np.zeros_like(x3_scaled_sol)
        for i in range(x3_scaled_sol.shape[1]):
            x3_sol[:,i] = self.eci.unscale_x(x3_scaled_sol[:,i]).full().flatten()
        u3_sol = np.zeros_like(u3_scaled_sol)
        for i in range(u3_scaled_sol.shape[1]):
            u3_sol[:,i] = self.eci.unscale_u(u3_scaled_sol[:,i]).full().flatten()
        if RecoveryStrategy != 'EXP':
            dt4_reentry_sol = dt4_reentry_scaled_sol * self.rocket_return.scaleT
            dt4_ballistic_sol = dt4_ballistic_scaled_sol * self.rocket_return.scaleT
            dt4_before_landing_sol = dt4_before_landing_scaled_sol * self.rocket_return.scaleT
            dt4_landing_sol = dt4_landing_scaled_sol * self.rocket_return.scaleT
            x4_before_reentry_sol = self.rocket_return.unscale_x(x4_before_reentry_scaled_sol).full().flatten()
            x4_after_reentry_sol = self.rocket_return.unscale_x(x4_after_reentry_scaled_sol).full().flatten()
            u4_reentry_sol = self.rocket_return.unscale_u([u4_reentry_scaled_sol]).full().flatten()
            x4_before_landing_sol = self.rocket_return.unscale_x(x4_before_landing_scaled_sol).full().flatten()
            x4_landing_sol  = np.zeros_like(x4_landing_scaled_sol)
            for i in range(x4_landing_scaled_sol.shape[1]):
                x4_landing_sol[:,i] = self.rocket_return.unscale_x(x4_landing_scaled_sol[:,i]).full().flatten()
            u4_landing_sol = np.zeros_like(u4_landing_scaled_sol)
            for i in range(u4_landing_scaled_sol.shape[0]):
                u4_landing_sol[i] = self.rocket_return.unscale_u([u4_landing_scaled_sol[i]]).full().flatten()[0]
            if RecoveryStrategy == 'RTLS':
                x4_0_sol = self.boostback.unscale_x(x4_0_scaled_sol).full().flatten()
                dt4_boostback_sol = dt4_boostback_scaled_sol * self.boostback.scaleT
                x4_boostback_sol = self.boostback.unscale_x(x4_boostback_scaled_sol).full().flatten()
                u4_boostback_sol = self.boostback.unscale_u(u4_boostback_scaled_sol).full().flatten()
            else:
                x4_0_sol = self.rocket_return.unscale_x(x4_0_scaled_sol).full().flatten()

        if 0:
            g_expr = opti.debug.g  # Symbolic expression for all constraints
            g_func = ca.Function("g_func", [opti.x], [g_expr])
            x_val = opti.debug.value(opti.x)
            g_val = g_func(x_val).full().flatten()

            lbg = opti.debug.lbg
            ubg = opti.debug.ubg # Evaluates g at solution

            tol = 1e-3
            print("Significant constraint residuals (> 1e-6):")
            for i, val in enumerate(g_val):
                lb = lbg[i]
                ub = ubg[i]
                if val < lb - tol or val > ub + tol:
                    # print(f"  g[{i}] = {val:.4e}, bounds = [{lb:.4e}, {ub:.4e}]")
                    print(f"  g[{i}] = {val:.4e}")
                    # print(g_expr[i])

            f_expr = opti.f  # symbolic expression for cost
            f_func = ca.Function("f_func", [opti.x], [f_expr])
            f_val = f_func(opti.debug.value(opti.x)).full().flatten()
            print(f"True objective value at solution: {f_val[0]:.6e}")

            g = opti.debug.g
            J = ca.jacobian(g, opti.x)
            J_func = ca.Function("J", [opti.x], [J])
            J_val = J_func(opti.debug.value(opti.x))
            print("Jacobian max:", np.max(np.abs(J_val.full())))

        # save the solution
        self.Solution = {}
        self.Solution['solution_units'] = 'SI'
        self.Solution['configuration'] = self.LV_Configuration
        self.Solution['launch_site'] = dict(self.launch_site)
        self.Solution['dispersion'] = dispersion_values
        self.Solution['u1'] = u1_sol
        self.Solution['x1'] = x1_sol
        self.Solution['dt1'] = dt1_sol
        self.Solution['u2'] = u2_sol
        self.Solution['x2'] = x2_sol
        self.Solution['dt2'] = dt2_sol
        self.Solution['u3'] = u3_sol
        self.Solution['x3'] = x3_sol
        self.Solution['dt3'] = dt3_sol
        self.Solution['payload_mass'] = payload_mass_sol
        self.Solution['propellant_mass_for_final_dv'] = float(propellent_mass_for_final_dv_sol)
        self.Solution['LaunchAz'] = LaunchAz_sol
        if RecoveryStrategy != 'EXP':
            self.Solution['x4_0'] = x4_0_sol
            self.Solution['dt4_reentry'] = dt4_reentry_sol
            self.Solution['dt4_ballistic'] = dt4_ballistic_sol
            self.Solution['dt4_before_landing'] = dt4_before_landing_sol
            self.Solution['dt4_landing'] = dt4_landing_sol
            self.Solution['x4_before_reentry'] = x4_before_reentry_sol
            self.Solution['x4_after_reentryburn'] = x4_after_reentry_sol
            self.Solution['u4_reentry'] = float(u4_reentry_sol[0])
            self.Solution['x4_before_landing'] = x4_before_landing_sol
            self.Solution['x4_landing'] = x4_landing_sol
            self.Solution['u4_landing'] = u4_landing_sol
            if RecoveryStrategy == 'RTLS':
                self.Solution['dt4_boostback'] = dt4_boostback_sol
                self.Solution['x4_boostback'] = x4_boostback_sol
                self.Solution['u4_boostback'] = u4_boostback_sol

        # detailed solution:
        t1_vec = np.linspace(init_time, init_time+self.booster.N*dt1_sol, self.booster.N+1)
        t2_vec = np.linspace(t1_vec[-1], t1_vec[-1]+self.eci.N[0]*dt2_sol, self.eci.N[0]+1) + self.eci.SecondStage_CoastTimeAfterSep
        t3_vec = np.linspace(t2_vec[-1], t2_vec[-1]+self.eci.N[1]*dt3_sol, self.eci.N[1]+1)

        Isp1_sol, acc1_sol = np.zeros((self.booster.N)), np.zeros((self.booster.N))
        Qdyn1_sol = np.zeros((self.booster.N+1))
        a1_sol, b1_sol = np.zeros((self.booster.N+1)), np.zeros((self.booster.N+1))
        apogee1_sol, perigee1_sol = np.zeros((self.booster.N+1)), np.zeros((self.booster.N+1))
        alt1_sol, vel1_sol = np.zeros((self.booster.N+1)), np.zeros((self.booster.N+1))
        i1_sol, gama1_sol, gama1_local_sol = np.zeros((self.booster.N+1)), np.zeros((self.booster.N+1)), np.zeros((self.booster.N+1))
        x1_eci_sol = np.zeros((self.eci.nx, self.booster.N+1))
        for k in range(self.booster.N+1):
            if k < self.booster.N:
                Isp1_sol[k] = self.booster.ISP_calc(x1_scaled_sol[:,k], u1_scaled_sol[:,k]).full().flatten()[0]
                acc1_sol[k] = self.booster.specific_acc_fun(x1_scaled_sol[:,k], u1_scaled_sol[:,k])
            Qdyn1_sol[k] = self.booster.dynamic_pressure_fun(x1_scaled_sol[:,k])
            pos, vel = self.eci.local_to_eci_func(ca.vertcat(x1_sol[0,k], 0.0, x1_sol[1,k]), ca.vertcat(x1_sol[2,k], 0.0, x1_sol[3,k]), LaunchAz_sol)
            x1_eci_sol[0:3,k], x1_eci_sol[3:6,k], x1_eci_sol[6,k] = pos.T, vel.T, x1_sol[4,k]
            alt1_sol[k] = np.linalg.norm(pos) - self.eci.R0 / np.sqrt(1.0 - self.eci.e**2 * np.sin(np.deg2rad(self.eci.LaunchLatitude))**2)
            vel1_sol[k] = np.linalg.norm(vel)
            h1 = ca.cross(pos, vel)
            e1 = ca.cross(vel, h1) / (self.eci.mu) - pos / np.linalg.norm(pos)
            eps1 = 0.5*ca.sumsqr(vel) - self.eci.mu / np.linalg.norm(pos)
            a1_sol[k] = -self.eci.mu / (2.0*eps1)
            b1_sol[k] = a1_sol[k] * np.sqrt(1.0 - ca.sumsqr(e1))
            apogee1_sol[k] = a1_sol[k] * (1.0 + np.linalg.norm(e1))
            perigee1_sol[k] = a1_sol[k] * (1.0 - np.linalg.norm(e1))
            i1_sol[k] = np.arccos(h1[2] / ca.norm_2(h1))
            gama1_sol[k] = np.arccos(ca.dot(vel, pos) / (ca.norm_2(vel) * ca.norm_2(pos)))
            gama1_local_sol[k] = np.arctan2(x1_sol[3,k], x1_sol[2,k])

        self.eci.local_to_eci_calc(ca.vertcat(x1_sol[0,k], 0.0, x1_sol[1,k]), ca.vertcat(x1_sol[2,k], 0.0, x1_sol[3,k]), LaunchAz_sol)
        Isp2_sol, acc2_sol = np.zeros((self.eci.N[0])), np.zeros((self.eci.N[0]))
        Alpha2_sol = np.zeros((self.eci.N[0]))
        a2_sol, b2_sol = np.zeros((self.eci.N[0]+1)), np.zeros((self.eci.N[0]+1))
        apogee2_sol, perigee2_sol = np.zeros((self.eci.N[0]+1)), np.zeros((self.eci.N[0]+1))
        alt2_sol, i2_sol = np.zeros((self.eci.N[0]+1)), np.zeros((self.eci.N[0]+1))
        Qdyn2_sol, gama2_sol = np.zeros((self.eci.N[0]+1)), np.zeros((self.eci.N[0]+1))
        for k in range(self.eci.N[0]+1):
            if k < self.eci.N[0]:
                Alpha2_sol[k] = np.arccos(np.dot(u2_sol[:,k], x2_sol[3:6,k]) / (self.eci.SecondStage_Thrust * ca.norm_2(x2_sol[3:6,k])))
                Isp2_sol[k] = self.eci.ISP_calc(x2_scaled_sol[:,k], u2_scaled_sol[:,k]).full().flatten()[0]
                acc2_sol[k] = self.eci.specific_acc_fun(x2_scaled_sol[:,k], u2_scaled_sol[:,k])
            pos, vel = x2_sol[0:3,k], x2_sol[3:6,k]
            alt2_sol[k] = np.linalg.norm(pos) - self.eci.R0 / np.sqrt(1.0 - self.eci.e**2 * np.sin(np.deg2rad(self.eci.LaunchLatitude))**2)
            Qdyn2_sol[k] = 0.5 * (x2_sol[3,k]**2 + x2_sol[4,k]**2 + x2_sol[5,k]**2) * self.atmosphere.rho_fun(alt2_sol[k])
            h2 = ca.cross(pos, vel)
            e2 = ca.cross(vel, h2) / (self.eci.mu) - pos / np.linalg.norm(pos)
            eps2 = 0.5*ca.sumsqr(vel) - self.eci.mu / np.linalg.norm(pos)
            a2_sol[k] = -self.eci.mu / (2.0*eps2)
            b2_sol[k] = a2_sol[k] * np.sqrt(1.0 - ca.sumsqr(e2))
            apogee2_sol[k] = a2_sol[k] * (1.0 + np.linalg.norm(e2))
            perigee2_sol[k] = a2_sol[k] * (1.0 - np.linalg.norm(e2))
            i2_sol[k] = np.arccos(h2[2] / ca.norm_2(h2))
            gama2_sol[k] = np.arccos(np.dot(vel, pos) / (ca.norm_2(vel) * ca.norm_2(pos)))

        Isp3_sol, acc3_sol = np.zeros((self.eci.N[1])), np.zeros((self.eci.N[1]))
        Alpha3_sol = np.zeros((self.eci.N[1]))
        a3_sol, b3_sol = np.zeros((self.eci.N[1]+1)), np.zeros((self.eci.N[1]+1))
        apogee3_sol, perigee3_sol = np.zeros((self.eci.N[1]+1)), np.zeros((self.eci.N[1]+1))
        alt3_sol, i3_sol, gama3_sol = np.zeros((self.eci.N[1]+1)), np.zeros((self.eci.N[1]+1)), np.zeros((self.eci.N[1]+1))
        for k in range(self.eci.N[1]+1):
            if k < self.eci.N[1]:
                Alpha3_sol[k] = np.arccos(np.dot(u3_sol[:,k], x3_sol[3:6,k]) / (self.eci.SecondStage_Thrust * ca.norm_2(x3_sol[3:6,k])))
                Isp3_sol[k] = self.eci.ISP_calc(x3_scaled_sol[:,k], u3_scaled_sol[:,k]).full().flatten()[0]
                acc3_sol[k] = self.eci.specific_acc_fun(x3_scaled_sol[:,k], u3_scaled_sol[:,k])
            pos, vel = x3_sol[0:3,k], x3_sol[3:6,k]
            alt3_sol[k] = np.linalg.norm(pos) - self.eci.R0 / np.sqrt(1.0 - self.eci.e**2 * np.sin(np.deg2rad(self.eci.LaunchLatitude))**2)
            h3 = ca.cross(pos, vel)
            e3 = ca.cross(vel, h3) / (self.eci.mu) - pos / np.linalg.norm(pos)
            eps3 = 0.5*ca.sumsqr(vel) - self.eci.mu / np.linalg.norm(pos)
            a3_sol[k] = -self.eci.mu / (2.0*eps3)
            b3_sol[k] = a3_sol[k] * np.sqrt(1.0 - ca.sumsqr(e3))
            apogee3_sol[k] = a3_sol[k] * (1.0 + np.linalg.norm(e3))
            perigee3_sol[k] = a3_sol[k] * (1.0 - np.linalg.norm(e3))
            i3_sol[k] = np.arccos(h3[2] / ca.norm_2(h3))
            gama3_sol[k] = np.arccos(np.dot(vel, pos) / (ca.norm_2(vel) * ca.norm_2(pos)))

        v3_apogee = ca.sqrt(self.eci.mu * (2.0 / apogee3_sol[-1] - 1.0 / a3_sol[-1]))
        v3_desired = ca.sqrt(self.eci.mu * (2.0 / apogee3_sol[-1] - 1.0 / Target_Orbit["a"]))
        propellent_mass_for_final_dv3 = x3_sol[6,self.eci.N[1]] * (1.0 - ca.exp((v3_apogee - v3_desired) / (self.eci.SecondStage_Vac_Isp * self.eci.g0)))
        actual_apogee3 = Target_Orbit["a"] * (1.0 + Target_Orbit["e"])

        if RecoveryStrategy != 'EXP':
            if RecoveryStrategy == 'RTLS':
                t4_boostback_vec = t1_vec[-1]+np.linspace(0, dt4_boostback_sol, self.rocket_return.N+1).reshape(-1,1)
                alt4_boostback_sol = np.zeros((self.rocket_return.N+1,1))
                vel4_boostback_sol = np.zeros((self.rocket_return.N+1,1))
                x4_boostback_vec_sol = np.zeros((self.boostback.nx, self.rocket_return.N+1))
                x4_boostback_vec_sol[:,0] = x4_0_sol
                Qdyn4_boostback_sol = np.zeros((self.rocket_return.N+1,1))
                acc4_boostback_sol = np.zeros((self.rocket_return.N+1,1))
                for k in range(self.rocket_return.N+1):
                    if k<self.rocket_return.N:
                        x4_boostback_vec_sol[:,k+1] = self.boostback.unscale_x(self.boostback.dynamics_kp1(self.boostback.scale_x(x4_boostback_vec_sol[:,k]), u4_boostback_scaled_sol, dt4_boostback_sol/self.rocket_return.N/self.boostback.scaleT)).full().flatten()
                    acc4_boostback_sol[k] = self.boostback.specific_acc_fun(self.boostback.scale_x(x4_boostback_vec_sol[:,k]), u4_boostback_scaled_sol)
                    alt4_boostback_sol[k] = ca.norm_2(ca.vertcat(x4_boostback_vec_sol[0,k], self.rocket_return.R0+x4_boostback_vec_sol[1,k])) - self.rocket_return.R0
                    vel4_boostback_sol[k] = ca.norm_2(ca.vertcat(x4_boostback_vec_sol[2,k], x4_boostback_vec_sol[3,k]))
                    Qdyn4_boostback_sol[k] = 0.5 * (self.atmosphere.rho_fun(alt4_boostback_sol[k]) * self.ScenarioDispersion.AtmosphereDensity) * vel4_boostback_sol[k]**2
            
            N4_ballistic = 25
            x4_ballistic_sol = np.zeros((self.rocket_return.nx, N4_ballistic+1))
            alt4_ballistic_sol, vel4_ballistic_sol = np.zeros((N4_ballistic+1,1)), np.zeros((N4_ballistic+1,1))
            Qdyn4_ballistic_sol = np.zeros((N4_ballistic+1,1))
            acc4_ballistic_sol = np.zeros((N4_ballistic+1,1))
            t4_ballistic_vec = np.zeros((N4_ballistic+1,1))
            if RecoveryStrategy == 'RTLS':
                x4_ballistic_sol[:,0] = self.boostback.unscale_x(x4_boostback_scaled_sol).full().flatten()
            else:
                x4_ballistic_sol[:,0] = x4_0_sol
            t4_ballistic_vec[0]  = t1_vec[-1] if RecoveryStrategy != 'RTLS' else dt4_boostback_sol+t1_vec[-1]
            for k in range(N4_ballistic+1):
                if k<N4_ballistic:
                    t4_ballistic_vec[k+1] = t4_ballistic_vec[k] + dt4_ballistic_sol/N4_ballistic
                    x4_ballistic_sol[:,k+1] = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1(self.rocket_return.scale_x(x4_ballistic_sol[:,k]), 0, dt4_ballistic_sol/N4_ballistic/self.rocket_return.scaleT)).full().flatten()
                acc4_ballistic_sol[k] = self.rocket_return.specific_acc_fun(self.rocket_return.scale_x(x4_ballistic_sol[:,k]), 0)
                alt4_ballistic_sol[k] = ca.norm_2(ca.vertcat(x4_ballistic_sol[0,k], self.rocket_return.R0+x4_ballistic_sol[1,k])) - self.rocket_return.R0
                vel4_ballistic_sol[k] = ca.norm_2(ca.vertcat(x4_ballistic_sol[2,k], x4_ballistic_sol[3,k]))
                Qdyn4_ballistic_sol[k] = 0.5 * (self.atmosphere.rho_fun(alt4_ballistic_sol[k]) * self.ScenarioDispersion.AtmosphereDensity) * vel4_ballistic_sol[k]**2

            x4_ballistic_M50_sol = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(x4_0_sol), 0, dt4_ballistic_sol/self.rocket_return.scaleT)).full().flatten()
            # before reentry burn phase
            alt4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[0], self.rocket_return.R0+x4_before_reentry_sol[1])) - self.rocket_return.R0
            vel4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[2], x4_before_reentry_sol[3]))
            Qdyn4_before_reentry_sol = 0.5 * (self.atmosphere.rho_fun(alt4_before_reentry_sol) * self.ScenarioDispersion.AtmosphereDensity) * vel4_before_reentry_sol**2

            # reentry phase
            N4_reentry = 25
            x4_reentry_sol = np.zeros((self.rocket_return.nx, N4_reentry+1))
            alt4_reentry_sol, vel4_reentry_sol = np.zeros((N4_reentry+1,1)), np.zeros((N4_reentry+1,1))
            Qdyn4_reentry_sol, hflux4_reentry_sol, acc4_reentry_sol = np.zeros((N4_reentry+1,1)), np.zeros((N4_reentry+1,1)), np.zeros((N4_reentry+1,1))
            x4_reentry_sol[:,0] = self.rocket_return.unscale_x(x4_before_reentry_scaled_sol).full().flatten()
            t4_reentry_vec = t4_ballistic_vec[-1] + np.linspace(0.0, dt4_reentry_sol, N4_reentry+1).reshape(-1,1)
            for k in range(N4_reentry+1):
                if k<N4_ballistic:
                    x4_reentry_sol[:,k+1] = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1(self.rocket_return.scale_x(x4_reentry_sol[:,k]), u4_reentry_scaled_sol, dt4_reentry_sol/N4_reentry/self.rocket_return.scaleT)).full().flatten()
                acc4_reentry_sol[k] = self.rocket_return.specific_acc_fun(self.rocket_return.scale_x(x4_reentry_sol[:,k]), u4_reentry_scaled_sol)
                alt4_reentry_sol[k] = ca.norm_2(ca.vertcat(x4_reentry_sol[0,k], self.rocket_return.R0+x4_reentry_sol[1,k])) - self.rocket_return.R0
                vel4_reentry_sol[k] = ca.norm_2(ca.vertcat(x4_reentry_sol[2,k], x4_reentry_sol[3,k]))
                Qdyn4_reentry_sol[k] = 0.5 * (self.atmosphere.rho_fun(alt4_reentry_sol[k]) * self.ScenarioDispersion.AtmosphereDensity) * vel4_reentry_sol[k]**2
                hflux4_reentry_sol[k] = self.rocket_return.Booster_k_empirical*np.sqrt((self.atmosphere.rho_fun(alt4_reentry_sol[k]) * self.ScenarioDispersion.AtmosphereDensity)) * vel4_reentry_sol[k]**3

            # before landing phase
            N4_before_landing = 25
            x4_before_landing_vec_sol = np.zeros((self.rocket_return.nx, N4_before_landing+1))
            alt4_before_landing_sol, vel4_before_landing_sol = np.zeros((N4_before_landing+1,1)), np.zeros((N4_before_landing+1,1))
            Qdyn4_before_landing_sol, hflux4_before_landing_sol, acc4_before_landing_sol = np.zeros((N4_before_landing+1,1)), np.zeros((N4_before_landing+1,1)), np.zeros((N4_before_landing+1,1))
            x4_before_landing_vec_sol[:,0] = self.rocket_return.unscale_x(x4_after_reentry_scaled_sol).full().flatten()
            t4_before_landing_vec = t4_reentry_vec[-1] + np.linspace(0.0, dt4_before_landing_sol, N4_before_landing+1).reshape(-1,1)
            for k in range(N4_before_landing+1):
                if k<N4_ballistic:
                    x4_before_landing_vec_sol[:,k+1] = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1(self.rocket_return.scale_x(x4_before_landing_vec_sol[:,k]), 0, dt4_before_landing_sol/N4_before_landing/self.rocket_return.scaleT)).full().flatten()
                acc4_before_landing_sol[k] = self.rocket_return.specific_acc_fun(self.rocket_return.scale_x(x4_before_landing_vec_sol[:,k]), 0)
                alt4_before_landing_sol[k] = self.rocket_return.local_to_alt(x4_before_landing_vec_sol[:,k])
                vel4_before_landing_sol[k] = ca.norm_2(ca.vertcat(x4_before_landing_vec_sol[2,k], x4_before_landing_vec_sol[3,k]))
                Qdyn4_before_landing_sol[k] = 0.5 * (self.atmosphere.rho_fun(alt4_before_landing_sol[k]) * self.ScenarioDispersion.AtmosphereDensity) * vel4_before_landing_sol[k]**2
                hflux4_before_landing_sol[k] = self.rocket_return.Booster_k_empirical * np.sqrt((self.atmosphere.rho_fun(alt4_before_landing_sol[k]) * self.ScenarioDispersion.AtmosphereDensity)) * vel4_before_landing_sol[k]**3

            # landing burn phase
            N4_landing = self.rocket_return.N
            alt4_landing_sol, vel4_landing_sol = np.zeros((N4_landing+1,1)), np.zeros((N4_landing+1,1))
            Qdyn4_landing_sol, acc4_landing_sol = np.zeros((N4_landing+1,1)), np.zeros((N4_landing+1,1))
            t4_landing_vec = t4_before_landing_vec[-1] + np.linspace(0, dt4_landing_sol*N4_landing, N4_landing+1).reshape(-1,1)
            for k in range(N4_landing+1):
                acc4_landing_sol[k] = self.rocket_return.specific_acc_fun(self.rocket_return.scale_x(x4_landing_sol[:,k]), u4_landing_scaled_sol[min(k,N4_landing-1)])
                alt4_landing_sol[k] = self.rocket_return.local_to_alt(x4_landing_sol[:,k])
                vel4_landing_sol[k] = ca.norm_2(ca.vertcat(x4_landing_sol[2,k], x4_landing_sol[3,k]))
                Qdyn4_landing_sol[k] = 0.5 * (self.atmosphere.rho_fun(alt4_landing_sol[k]) * self.ScenarioDispersion.AtmosphereDensity) * vel4_landing_sol[k]**2

        self.Solution['t1_vec'] = t1_vec
        self.Solution['t2_vec'] = t2_vec
        self.Solution['t3_vec'] = t3_vec
        self.Solution['alt1_sol'] = alt1_sol
        self.Solution['Qdyn1_sol'] = Qdyn1_sol
        self.Solution['acc1_sol'] = acc1_sol
        self.Solution['Isp1_sol'] = Isp1_sol
        self.Solution['alt2_sol'] = alt2_sol
        self.Solution['Qdyn2_sol'] = Qdyn2_sol
        self.Solution['Alpha2_sol'] = Alpha2_sol
        self.Solution['acc2_sol'] = acc2_sol    
        self.Solution['i2_sol'] = i2_sol
        self.Solution['apogee2_sol'] = apogee2_sol
        self.Solution['perigee2_sol'] = perigee2_sol
        self.Solution['Isp2_sol'] = Isp2_sol
        self.Solution['alt3_sol'] = alt3_sol
        self.Solution['acc3_sol'] = acc3_sol
        self.Solution['apogee3_sol'] = apogee3_sol
        self.Solution['perigee3_sol'] = perigee3_sol
        self.Solution['i3_sol'] = i3_sol
        self.Solution['Isp3_sol'] = Isp3_sol
        self.Solution['Alpha3_sol'] = Alpha3_sol

        if RecoveryStrategy != 'EXP':
            if RecoveryStrategy == 'RTLS':
                t4_vec = np.vstack((t4_boostback_vec, t4_ballistic_vec, t4_reentry_vec, t4_before_landing_vec, t4_landing_vec))
                x4 = np.hstack((x4_boostback_vec_sol, x4_ballistic_sol, x4_reentry_sol, x4_before_landing_vec_sol, x4_landing_sol)).T
                alt4_sol = np.vstack((alt4_boostback_sol, alt4_ballistic_sol, alt4_reentry_sol, alt4_before_landing_sol, alt4_landing_sol))
                Qdyn4_sol = np.vstack((Qdyn4_boostback_sol, Qdyn4_ballistic_sol, Qdyn4_reentry_sol, Qdyn4_before_landing_sol, Qdyn4_landing_sol))
                acc4_sol = np.vstack((acc4_boostback_sol, acc4_ballistic_sol, acc4_reentry_sol, acc4_before_landing_sol, acc4_landing_sol))
            else:
                t4_vec = np.vstack((t4_ballistic_vec, t4_reentry_vec, t4_before_landing_vec, t4_landing_vec))
                x4 = np.vstack((x4_ballistic_sol.T, x4_reentry_sol.T, x4_before_landing_vec_sol.T, x4_landing_sol.T))
                alt4_sol = np.vstack((alt4_ballistic_sol, alt4_reentry_sol, alt4_before_landing_sol, alt4_landing_sol))
                Qdyn4_sol = np.vstack((Qdyn4_ballistic_sol, Qdyn4_reentry_sol, Qdyn4_before_landing_sol, Qdyn4_landing_sol))
                acc4_sol = np.vstack((acc4_ballistic_sol, acc4_reentry_sol, acc4_before_landing_sol, acc4_landing_sol))
            self.Solution['t4_vec'] = t4_vec
            self.Solution['x4'] = x4
            self.Solution['alt4_sol'] = alt4_sol
            self.Solution['Qdyn4_sol'] = Qdyn4_sol
            self.Solution['acc4_sol'] = acc4_sol

        self.Solution['v3_desired'] = v3_desired
        self.Solution['return_status'] = opti.return_status()
        self.Solution['success'] = opti.stats()['success']
        self.Solution['iter_count'] = opti.stats().get('iter_count')
        if solve_error is not None:
            self.Solution['success'] = False
            self.Solution['solver_error'] = solve_error

        # if output_dir is not None:
        #     result_directory = Path(output_dir)
        #     result_directory.mkdir(parents=True, exist_ok=True)
        #     with (result_directory / filename).open('wb') as f:
        #         pickle.dump(self.Solution, f)

        return self.Solution
