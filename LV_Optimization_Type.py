#!/usr/bin/env python3

from dataclasses import dataclass, field
import numpy as np
import casadi as ca
import Atmosphere_Type as Atm_Type
import LV_Type_scaled as LV_Type
import LV_Plot as LVp
import pickle

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

@dataclass
class LV_Optimization(VLType):
    def __init__(self, ScenarioDispersion: LV_Type.DispesrionFactorsType, 
                         LV_Configuration: int = 1):
        super().__init__()

        self.ScenarioDispersion = ScenarioDispersion
        self.LV_Configuration = LV_Configuration

        # Initialize Atmosphere:
        self.atmosphere = Atm_Type.AtmosphereType()

        # Initialize the rocket models
        self.booster = LV_Type.BoosterLaunchVehicle_2D(self.atmosphere, LV_Configuration, ScenarioDispersion)
        self.boostback = LV_Type.BoostBackBurn_2D(ScenarioDispersion, LV_Configuration)
        self.eci = LV_Type.LaunchVehicle_ECI(ScenarioDispersion, LV_Configuration)
        self.rocket_return = LV_Type.BoosterReturn_2D(self.atmosphere, LV_Configuration, ScenarioDispersion)

    def SolveOptimiztion(self, RecoveryStrategy: list = 'EXP',
                         payload_mass_predefined: float = -1.0, 
                         Target_Orbit = dict, 
                         Plot_interm: bool = False, 
                         init_guess = None):
        
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

        # First Phase - Booster 
        if payload_mass_predefined > 0.0:
            payload_mass_scaled = payload_mass_predefined / self.booster.scaleX[4]
        else:
            payload_mass_scaled = opti.variable(1, 1)
            # opti.subject_to(payload_mass_scaled >= 0.0)
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

        opti.subject_to(x1[:,0] == self.booster.scale_x(ca.vertcat(init_x, init_z, init_vx, init_vz, init_m)))
        opti.subject_to(dt1 >= 0.1/self.booster.scaleT)
        # set the constraints:
        for k in range(self.booster.N):
            # set the dynamics
            opti.subject_to(x1[:,k+1] == self.booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))
            # set the thrust constraints
            opti.subject_to(u1[0,k] <= 1.0)
            opti.subject_to(u1[0,k] >= self.booster.FirstStage_MinThrust_Factor)
            opti.subject_to(u1[1,k]**2 <= 1.0)

            # set the state constraints
            uk = self.booster.unscale_u(u1[:,k])
            q = self.booster.dynamic_pressure_fun(x1[:,k])

            opti.subject_to(q/self.booster.FirstStage_MaxDynamicPressure <= 1.0) # dynamic pressure limit
            opti.subject_to((uk[1] * q / 1e4)**2 <= 1.0) # alpha * dynamic pressure limit
            opti.subject_to(self.booster.specific_acc_fun(x1[:,k], u1[:,k])/self.booster.Payload_Max_acc <= 1.0)

        # set the final state constraints    
        x1f = self.booster.unscale_x(x1[:,self.booster.N])
        q1f = 0.5 * (x1f[2]**2 + x1f[3]**2) * self.atmosphere.rho_fun(self.booster.local_to_alt(x1f))
        opti.subject_to(q1f/self.booster.FirstStage_StageSeparationMaxDynamicPressure <= 1.0)
        if RecoveryStrategy == 'EXP':
            opti.subject_to(x1f[4] >= self.eci.EmptyFirstStageMass + self.eci.SecondStage_FullMass + payload_mass)

        # Phase 2 - Second Stage, before fairing separation (if needed)
        x2 = opti.variable(self.eci.nx, self.eci.N[0]+1) # [x, y, z, vx, vy, vz, m]
        u2 = opti.variable(self.eci.nu, self.eci.N[0]) # [fx, fy, fz]
        dt2 = opti.variable(1)

        # Set the initial state - the final state of the boost phase transformed to ECI
        opti.subject_to(dt2 >= 0.1/self.eci.scaleT)
        opti.subject_to(LaunchAz >= -np.pi/2)
        opti.subject_to(LaunchAz <= np.pi/2)
        eci_pos, eci_vel = self.eci.local_to_eci_func(ca.vertcat(x1f[0], 0.0, x1f[1]), ca.vertcat(x1f[2], 0.0, x1f[3]), LaunchAz)
        x2_init = ca.vertcat(eci_pos, eci_vel, self.eci.SecondStage_FullMass + payload_mass)
        opti.subject_to(x2[:,0] == self.eci.scale_x(x2_init))
        for k in range(self.eci.N[0]):
            # set the dynamics
            opti.subject_to(x2[:,k+1] == self.eci.dynamics_kp1(x2[:,k], u2[:,k], dt2))
            # set the thrust constraints
            opti.subject_to(ca.norm_2(u2[:,k]) <= 1.0)
            opti.subject_to(ca.norm_2(u2[:,k]) >= self.eci.SecondStage_MinThrust_Factor)

        # set the final state constraints
        opti.subject_to(self.eci.eci_to_alt(self.eci.unscale_x(x2[:,self.eci.N[0]]))/self.eci.FairingSeparationAltitude >= 1.0)

        ## phase 3 - Second Stage, after fairing separation
        x3 = opti.variable(self.eci.nx, self.eci.N[1]+1) # [x, y, z, vx, vy, vz, m]
        u3 = opti.variable(self.eci.nu, self.eci.N[1])
        dt3 = opti.variable(1)
        # Set the initial state - the final state of the boost phase transformed to ECI
        x3_init = ca.vertcat(x2[0:6,self.eci.N[0]], x2[6,self.eci.N[0]] - self.eci.FairingMass/ self.eci.scaleX[6])
        opti.subject_to(dt3 >= 0.1/self.eci.scaleT)
        opti.subject_to(x3[:,0] == x3_init)
        for k in range(self.eci.N[1]):
            # set the dynamics
            opti.subject_to(x3[:,k+1] == self.eci.dynamics_kp1(x3[:,k], u3[:,k], dt3))
            # set the thrust constraints
            opti.subject_to(ca.norm_2(u3[:,k]) <= 1.0)
            opti.subject_to(ca.norm_2(u3[:,k]) >= self.eci.SecondStage_MinThrust_Factor)
            opti.subject_to(self.eci.specific_acc_fun(x3[:,k], u3[:,k])/self.booster.Payload_Max_acc <= 1.0)

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
        propellent_mass_for_final_dv = x3f[6] * (1.0 - ca.exp(-ca.fabs(v_apogee - v_desired) / (self.eci.SecondStage_Vac_Isp * self.eci.g0)))
        
        # v_final_apogee = ca.sqrt(self.rocket_eci.mu * (1.0 / Target_Orbit["a"]))
        # dv_for_inclination_fix = 2 * v_apogee * ca.sin(Target_Orbit["i"]/2.0)
        # propellent_mass_for_inclination_fix = (x3[6,self.rocket_eci.N[1]]-propellent_mass_for_final_dv) * (1.0 - ca.exp(-ca.fabs(dv_for_inclination_fix) / (self.rocket_eci.SecondStage_Vac_Isp * self.rocket_eci.g0)))


        # propellent_mass_for_final_dv = 0.0
        x3f_mass_scaled = (self.eci.SecondStage_EmptyMass + payload_mass + propellent_mass_for_final_dv) / self.eci.scaleX[6]
        opti.subject_to(x3[6,self.eci.N[1]] >= x3f_mass_scaled)
        opti.subject_to(apogee/Target_Orbit["perigee"] == 1.0)
        opti.subject_to(perigee/(self.eci.R0 + self.eci.ParkingOrbit_PerigeeAlt) >= 1.0)
        opti.subject_to(i/Target_Orbit["i"] == 1.0)


    # Phase 4 - Booster Return:
        if RecoveryStrategy != 'EXP':
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
                opti.subject_to(dt4_boostback >= 1.0 / self.boostback.scaleT)
            opti.subject_to(dt4_ballistic >= 75.0 / self.rocket_return.scaleT)
            opti.subject_to(dt4_reentry >= 0.1 / self.rocket_return.scaleT)
            opti.subject_to(dt4_before_landing >= 0.1 / self.rocket_return.scaleT)
            opti.subject_to(dt4_landing >= 0.1 / self.rocket_return.scaleT)

            # Set the initial state
            x4_init_unscaled = ca.vertcat(x1f[0:4], x1f[4] - (self.eci.SecondStage_FullMass + payload_mass))
            
            # Boostback phase and ballistic phase
            if RecoveryStrategy == 'RTLS':
                opti.subject_to(x4_0 == self.boostback.scale_x(x4_init_unscaled))
                opti.subject_to(x4_boostback == self.boostback.dynamics_kp1_M20(x4_0, u4_boostback, dt4_boostback))
                opti.subject_to(ca.norm_2(u4_boostback) <= 1.0)

                opti.subject_to(x4_before_reentry == self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(self.boostback.unscale_x(x4_boostback)), 0, dt4_ballistic))
            else:
                opti.subject_to(x4_0 == self.rocket_return.scale_x(x4_init_unscaled))
                opti.subject_to(x4_before_reentry == self.rocket_return.dynamics_kp1_M50(x4_0, 0, dt4_ballistic))
            opti.subject_to(self.rocket_return.local_to_alt(self.rocket_return.unscale_x(x4_before_reentry))/50000.0 >= 1.0)

            # rentry burn
            opti.subject_to(x4_after_reentryburn  == self.rocket_return.dynamics_kp1_M20(x4_before_reentry, u4_reentry, dt4_reentry))
            ### set the thrust constraints
            opti.subject_to(u4_reentry[0] <= 1.0)
            opti.subject_to(u4_reentry[0] >= self.rocket_return.FirstStage_MinThrust_Factor)

            ### After Reentry, before landing burn (ballistic)
            opti.subject_to(x4_before_landing == self.rocket_return.dynamics_kp1_M20(x4_after_reentryburn, 0, dt4_before_landing))

            N_before_landing_constraints = 20
            x4_before_landing_intren = x4_after_reentryburn
            for i in range(N_before_landing_constraints):
                x4_before_landing_intren = self.rocket_return.dynamics_kp1(x4_before_landing_intren, 0, dt4_before_landing/N_before_landing_constraints)
                opti.subject_to(self.rocket_return.dynamic_pressure_fun(x4_before_landing_intren)/self.rocket_return.Booster_MaxDynamicPressure <= 1.0)
                opti.subject_to(self.rocket_return.heat_flux_fun(x4_before_landing_intren)/self.rocket_return.Booster_MaxHeatFlux <= 1.0)

            ### Landing Burn:
            opti.subject_to(x4_landing[:,0] == x4_before_landing)
            for k in range(self.rocket_return.N):
                opti.subject_to(x4_landing[:,k+1] == self.rocket_return.dynamics_kp1(x4_landing[:,k], u4_landing[:,k], dt4_landing))
                opti.subject_to(u4_landing[0,k] <= 1.0/3.0) 
                opti.subject_to(u4_landing[0,k] >= 0.75*1.0/3.0*self.rocket_return.EmptyFirstStageMass*self.rocket_return.g0/self.rocket_return.scaleU[0])
                opti.subject_to(self.rocket_return.dynamic_pressure_fun(x4_landing[:,k+1])/self.rocket_return.Booster_MaxDynamicPressure <= 1.0)
                opti.subject_to(self.rocket_return.heat_flux_fun(x4_landing[:,k+1])/self.rocket_return.Booster_MaxHeatFlux <= 1.0)

            # final state constraints
            x4f_landing_unscaled = self.rocket_return.unscale_x(x4_landing[:,self.rocket_return.N])
            if RecoveryStrategy == 'RTLS':
                # opti.subject_to(ca.norm_2(x4_landing[0:2,self.rocket_return.N]) == 0.0)
                opti.subject_to(x4_landing[0,self.rocket_return.N] == 0.0)
                opti.subject_to(x4_landing[1,self.rocket_return.N] == 0.0)
            else:
                # zero altitude at the end of the flight
                opti.subject_to(self.rocket_return.local_to_alt(x4f_landing_unscaled) == 0.0) # landing altitude
            # minimal velocity at the end of the flight
            opti.subject_to(ca.norm_2(x4_landing[2:4,self.rocket_return.N]) <= 10/self.rocket_return.scaleX[2]) # minimal velocity at the end of the flight
            opti.subject_to(x4_landing[4,self.rocket_return.N] >= (self.rocket_return.EmptyFirstStageMass)/self.rocket_return.scaleX[4]) # minimal mass at the end of the flight

        # set the cost function
        gain = 1e-2
        cost = 0.0
        cost += 0.5*gain*(ca.sumsqr(u1[0,1:]-u1[0,:-1]) + ca.sumsqr(u1[1,1:]-u1[1,:-1])) 
        cost += 0.5*gain*(ca.sumsqr(u2[0,1:]-u2[0,:-1])  + ca.sumsqr(u3[1,1:]-u3[1,:-1]) )
        if RecoveryStrategy != 'EXP':
            cost += 0.5*gain*ca.sumsqr(u4_landing[1:]-u4_landing[:-1]) * dt4_landing
        cost += -payload_mass_scaled*100
        if type(payload_mass_scaled) != ca.MX:
            cost += -x3f[6]
        opti.minimize(cost)


        # set the solver
        
        opts = {"print_time": 0,  # Print timing, 
                "ipopt": {
                "linear_solver": "ma97", "hsllib": "/usr/local/lib/libcoinhsl.so",  # MA97 solver Path to HSL library
                "mu_strategy": "adaptive",  # "adaptive" or "adaptive" Strategy for updating the barrier parameter
                "tol": 1e-6,  # Convergence tolerance
                "max_iter": 250,  # Max iterations
                "print_level": 0,  # Verbosity level
                'print_frequency_iter': 5,  # print_frequency_iter
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
            v3_0_guess = 2400.0
            v3_0_guess = 1750.0
        else:
            v3_0_guess = 2700.0

        if payload_mass_predefined <= 0.0:
            alpha = ca.exp(ca.fabs(v_parking-v3_0_guess) / (self.eci.SecondStage_Vac_Isp * self.eci.g0))
            if type(init_guess) == float:
                payload_mass_guess = init_guess*(self.eci.SecondStage_FullMass - alpha*self.eci.SecondStage_EmptyMass)/(alpha-1.0)
            else:
                payload_mass_guess = init_guess['payload_mass']
        else:
            payload_mass_guess = payload_mass_predefined*0.25
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
                if np.atan2(xk[3], xk[2]) > 87.0*np.pi/180.0 and np.linalg.norm(xk[2:4]) > 75.0:
                    u1_guess[1,k] = -np.deg2rad(1.5) / self.booster.FirstStage_MaxAlpha
                else:
                    u1_guess[1,k] = 0.0


                max_u0 = self.booster.Payload_Max_acc*xk[4]/self.booster.FirstStage_Vac_Thrust
                u1_guess[0,k] = np.clip(u1_guess[0,k], self.booster.FirstStage_MinThrust_Factor, max_u0)
                
                x1_guess[:,k+1] = self.booster.dynamics_kp1(x1_guess[:,k], u1_guess[:,k], dt1_guess).full().flatten()
                acc = self.booster.specific_acc_fun(x1_guess[:,k], u1_guess[:,k])

            xN = self.booster.unscale_x(x1_guess[:,self.booster.N])
            alt_N1 = self.booster.local_to_alt(xN)
            Q = 0.5 * (xN[2]**2 + xN[3]**2) * self.atmosphere.rho_fun(alt_N1)
            if Q > self.booster.FirstStage_StageSeparationMaxDynamicPressure:
                dt1_guess += 0.5 / self.booster.N / self.booster.scaleT
                iter += 1
            elif xN[4] <= self.booster.FirstStage_EmptyMass + payload_mass_guess + dm:
                dt1_guess -= 1.0 / self.booster.N / self.booster.scaleT
                iter += 1
            else:
                break

        # calculate the initial guess for second stage before fairing separation, if needed
        dt2_guess = 50.0 / self.eci.N[0] / self.eci.scaleT
        x2_guess = np.zeros((self.eci.nx, self.eci.N[0]+1))
        u2_guess = np.zeros((self.eci.nu, self.eci.N[0]))
        x1f_guess = self.booster.unscale_x(x1_guess[:,self.booster.N])
        pos, vel = self.eci.local_to_eci_func(ca.vertcat(x1f_guess[0], 0.0, x1f_guess[1]), ca.vertcat(x1f_guess[2], 0.0, x1f_guess[3]), LaunchAz_guess)
        x2_init = ca.vertcat(pos, vel, self.eci.SecondStage_FullMass + payload_mass_guess)
        x2_guess[:,0] = self.eci.scale_x(x2_init).full().flatten()
        iter=0
        while iter < 10:
            for k in range(self.eci.N[0]):
                xk = self.eci.unscale_x(x2_guess[:,k])
                alt = self.eci.eci_to_alt(xk)
                u2_guess[0:3,k] = (x2_guess[3:6,k] / ca.norm_2(x2_guess[3:6,k])).full().flatten()
                x2_guess[:,k+1] = self.eci.dynamics_kp1(x2_guess[:,k], u2_guess[:,k], dt2_guess).full().flatten()
            xN = self.eci.unscale_x(x2_guess[:,self.eci.N[0]])
            alt = self.eci.eci_to_alt(xN)
            if abs(alt-self.eci.FairingSeparationAltitude) < 100.0:
                break
            else:
                dt2_guess = dt2_guess*(self.eci.FairingSeparationAltitude-alt_N1)/ (alt-alt_N1)
                iter += 1

    # calculate the initial guess for second stage after fairing separation
        dt3_guess = 370.0/self.eci.N[1]/self.eci.scaleT
        x3_guess = np.zeros((self.eci.nx, self.eci.N[1]+1))
        u3_guess = np.zeros((self.eci.nu, self.eci.N[1]))
        x3_guess[0:6,0] = x2_guess[0:6,self.eci.N[0]]
        x3_guess[6,0] = x2_guess[6,self.eci.N[0]] - self.eci.FairingMass / self.eci.scaleX[6]
        iter=0

        alpha3_init = 0.0
        v3_max = ca.sqrt(self.eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["apogee"]))
        while iter < 50:
            alpha3, alpha_factor = alpha3_init, 1.0
            for k in range(self.eci.N[1]):
                xk = self.eci.unscale_x(x3_guess[:,k]).full().flatten()
                h = np.cross(xk[0:3], xk[3:6])
                h = h / np.linalg.norm(h)
                u3_guess[0:3,k] = (xk[3:6] / np.linalg.norm(xk[3:6])) 
                # alpha_factor = k/self.rocket_eci.N[1]
                alpha3 = -alpha3_init*alpha_factor
                u3_guess[0:3,k] = u3_guess[0:3,k] * np.cos(alpha3) + np.cross(h, u3_guess[0:3,k]) * np.sin(alpha3) + h * np.dot(h, u3_guess[0:3,k]) * (1 - np.cos(alpha3)) 
                acc = self.eci.SecondStage_Thrust / xk[6]
                if acc > self.eci.Payload_Max_acc:
                    u3_guess[0:3,k] = u3_guess[0:3,k] * self.eci.Payload_Max_acc/acc
                x3_guess[:,k+1] = self.eci.dynamics_kp1(x3_guess[:,k], u3_guess[:,k], dt3_guess).full().flatten()
            
                xkp1 = self.eci.unscale_x(x3_guess[:,k+1]).full().flatten()
                vel = np.linalg.norm(xkp1[3:6])
                h_guess = np.cross(xkp1[0:3], xkp1[3:6])
                e_guess = ca.cross(xkp1[3:6], h_guess) / (self.eci.mu) - xkp1[0:3] / ca.norm_2(xkp1[0:3])
                eps_guess = 0.5*ca.sumsqr(xkp1[3:6]) - self.eci.mu / ca.norm_2(xkp1[0:3])
                a_guess = -self.eci.mu / (2.0*eps_guess)
                perigee_guess = a_guess * (1.0 - ca.norm_2(e_guess))
                apogee_guess = a_guess * (1.0 + ca.norm_2(e_guess))
                i_guess = ca.acos(h_guess[2] / ca.norm_2(h_guess))


                
                if xkp1[6] < m3_final + self.eci.SecondStage_EmptyMass + payload_mass_guess:
                    break
            if k < self.eci.N[1]-1:
                iter += 1
                dt3_guess = dt3_guess * (k+1) / self.eci.N[1]
            elif xkp1[6] < m3_final:
                dt3_guess -= 0.25/self.eci.scaleT
                iter += 1
            elif apogee_guess < self.eci.ParkingOrbit_PerigeeAlt + self.eci.R0 or a_guess < 0.5*(Target_Orbit['perigee']+self.eci.R0+200.0*1e3):
                dt3_guess += 0.1/self.eci.scaleT
                iter += 1
            elif perigee_guess < self.eci.R0+10*10**3:
                alpha3_init -= np.deg2rad(0.5)
                # dt3_guess += 0.
                iter += 1
            elif apogee_guess > Target_Orbit['perigee'] + 150.0*1e3:
                dt3_guess -= 0.1/self.eci.scaleT
                iter += 1
            else:
                break


        # calculate the initial guess for self.booster return
        if RecoveryStrategy != 'EXP':
            # calculate the initial guess for self.booster return
            x4_0_guess = np.zeros((self.rocket_return.nx, 1))
            x1f_guess_unscaled = self.booster.unscale_x(x1_guess[:,self.booster.N]).full().flatten()
            x4_0_guess[0:4,0] = x1f_guess_unscaled[0:4]
            x4_0_guess[4,0] = x1f_guess_unscaled[4] - self.eci.SecondStage_FullMass - payload_mass_guess
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
                    if abs(x_bal) > 100.0:
                        dt4_boostback_guess += np.clip(x_bal / abs(x4_boostback_guess_unscaled[2])/400, -1.0, 1.0) / self.boostback.scaleT
                        iter += 1
                    else:
                        break

            # calculate the initial guess for self.booster return
            dt4_ballistic_guess = 300 / self.rocket_return.scaleT
            iter=0
            while iter < 50:
                if RecoveryStrategy == 'RTLS':
                    x4_before_reentry_guess = self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(x4_boostback_guess_unscaled), 0, dt4_ballistic_guess)
                else:
                    x4_before_reentry_guess = self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(x4_0_guess), 0, dt4_ballistic_guess)
                x4_before_reentry_guess_unscaled = self.rocket_return.unscale_x(x4_before_reentry_guess)
                alt = self.rocket_return.local_to_alt(x4_before_reentry_guess_unscaled)
                vel = ca.norm_2(x4_before_reentry_guess_unscaled[2:4,0])
                if abs(alt - 60000) > 10:
                    dt4_ballistic_guess += np.clip((alt - 60000)/abs(x4_before_reentry_guess_unscaled[3,0]), -10.0, 10.0) / self.rocket_return.scaleT
                    iter += 1
                else:
                    break
            
            u4_reentry_guess = 1.0
            dt4_reentry_guess = 1.0 / self.rocket_return.scaleT
            iter = 0
            v_final = 1300.0
            while iter < 50:
                x4_after_reentry_guess = self.rocket_return.dynamics_kp1_M20(x4_before_reentry_guess, u4_reentry_guess, dt4_reentry_guess)
                x4_after_reentry_guess_unscaled = self.rocket_return.unscale_x(x4_after_reentry_guess)
                alt = self.rocket_return.local_to_alt(x4_after_reentry_guess_unscaled)
                vel = ca.norm_2(x4_after_reentry_guess_unscaled[2:4,0])
                mass = x4_after_reentry_guess_unscaled[4,0]
                if abs(vel- v_final) > 10:
                    dt4_reentry_guess += 0.5 * np.sign(vel - v_final) / self.rocket_return.scaleT
                else:
                    break
                iter += 1

            # after reentry burn before landing burn
            iter = 0
            dt4_before_landing_guess = 50.0 / self.rocket_return.scaleT
            alt4_before_landing = 1000.0
            while iter < 50:
                x4_before_landing_guess = self.rocket_return.dynamics_kp1_M20(x4_after_reentry_guess, 0, dt4_before_landing_guess)
                x4_before_landing_guess_unscaled = self.rocket_return.unscale_x(x4_before_landing_guess).full().flatten()
                alt = self.rocket_return.local_to_alt(x4_before_landing_guess_unscaled)
                vel = np.linalg.norm(x4_before_landing_guess_unscaled[2:4])
                if abs(alt - alt4_before_landing) > 10:
                    dt4_before_landing_guess += 0.9*(alt - alt4_before_landing)/vel / self.rocket_return.scaleT
                    iter += 1
                else:
                    break

            # initial guess for the landing burn
            vel = np.linalg.norm(x4_before_landing_guess_unscaled[2:4])
            alt  = self.rocket_return.local_to_alt(x4_before_landing_guess_unscaled)
            dt4_landing_guess = vel/(self.rocket_return.scaleU[0]/x4_before_landing_guess_unscaled[4]/3-self.rocket_return.g0) / self.rocket_return.N / self.rocket_return.scaleT
            x4_landing_guess = np.zeros((self.rocket_return.nx, self.rocket_return.N+1))
            u4_landing_guess = np.zeros((self.rocket_return.nu, self.rocket_return.N))
            iter = 0
            thrust_factor = 1.0
            while iter < 20:
                x4_landing_guess[:,0] = x4_before_landing_guess.full().flatten()
                for k in range(self.rocket_return.N):
                    xk = self.rocket_return.unscale_x(x4_landing_guess[:,k]).full().flatten()
                    vel = np.linalg.norm(xk[2:4])
                    alt = self.rocket_return.local_to_alt(xk)
                    g_vec = -np.array([xk[0], self.rocket_return.R0 + xk[1]])
                    g_vec /= np.linalg.norm(g_vec)
                    g = self.rocket_return.g0 * (self.rocket_return.R0/(self.rocket_return.R0 + xk[1]))**2
                    g_v = (g_vec[0]*xk[2] + g_vec[1]*xk[3])/vel * g
                    u4_landing_guess[0,k] = g_v * xk[4]/self.rocket_return.scaleU[0]                    
                    if vel > 30.0 and alt > 50.0:
                        u4_landing_guess[0,k] += (3*vel**2)/(2*alt)*xk[4]/self.rocket_return.scaleU[0]
                    if alt < 50.0:
                        u4_landing_guess[0,k] += 0.5*(vel-0.5)**2/alt*xk[4]/self.rocket_return.scaleU[0]
                    u4_landing_guess[0,k] = thrust_factor*np.clip(u4_landing_guess[0,k], 0.01, 1.0/3.0)
                    x4_landing_guess[:,k+1] = self.rocket_return.dynamics_kp1(x4_landing_guess[:,k], u4_landing_guess[:,k], dt4_landing_guess).full().flatten()
                    xkp1 = self.rocket_return.unscale_x(x4_landing_guess[:,k+1]).full().flatten()
                    alt = self.rocket_return.local_to_alt(xkp1)
                    vel = np.linalg.norm(xkp1[2:4])
                    if alt < -1.0:
                        break
                if alt > 10:
                    dt4_landing_guess += (alt)/vel/self.rocket_return.N / self.rocket_return.scaleT
                    # thrust_factor *= 1.01
                    iter += 1
                # elif vel > 20:
                #     thrust_factor += self.rocket_return.FirstStage_EmptyMass/self.rocket_return.FirstStage_SL_Thrust
                #     iter += 1
                elif alt < -1.0 and k<self.rocket_return.N-1:
                    dt4_landing_guess *= k/(self.rocket_return.N-1) / self.rocket_return.scaleT
                    iter += 1
                elif alt < -1.0:
                    thrust_factor *= 1.1
                    iter += 1
                else:
                    break


        # set the initial guess for the optimization variables
        if type(payload_mass_scaled) == ca.MX:
            opti.set_initial(payload_mass_scaled, payload_mass_guess/self.booster.scaleX[4])
        opti.set_initial(dt1, dt1_guess)
        opti.set_initial(x1, x1_guess)
        opti.set_initial(u1, u1_guess)
        opti.set_initial(LaunchAz, LaunchAz_guess)
        opti.set_initial(dt2, dt2_guess)
        opti.set_initial(x2, x2_guess)
        opti.set_initial(u2, u2_guess)
        opti.set_initial(dt3, dt3_guess)
        opti.set_initial(x3, x3_guess)
        opti.set_initial(u3, u3_guess)
        if RecoveryStrategy != 'EXP':
            opti.set_initial(dt4_ballistic, dt4_ballistic_guess)
            opti.set_initial(dt4_reentry, dt4_reentry_guess)
            opti.set_initial(dt4_before_landing, dt4_before_landing_guess)
            opti.set_initial(dt4_landing, dt4_landing_guess)
            opti.set_initial(x4_before_reentry, x4_before_reentry_guess)
            opti.set_initial(x4_after_reentryburn, x4_after_reentry_guess)
            opti.set_initial(u4_reentry, u4_reentry_guess)
            opti.set_initial(x4_before_landing, x4_before_landing_guess)
            opti.set_initial(x4_landing, x4_landing_guess)
            opti.set_initial(u4_landing, u4_landing_guess)
            if RecoveryStrategy == 'RTLS':
                opti.set_initial(x4_0, self.boostback.scale_x(x4_0_guess))
                opti.set_initial(dt4_boostback, dt4_boostback_guess)
                opti.set_initial(x4_boostback, x4_boostback_guess)
                opti.set_initial(u4_boostback, u4_boostback_guess)
            else:
                opti.set_initial(x4_0, self.rocket_return.scale_x(x4_0_guess))

        if type(init_guess) == dict:
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
                opti.set_initial(u4_reentry, self.rocket_return.scale_u([init_guess['u4_reentry']]))
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


            filename = f"LV_{RecoveryStrategy}_apogee_{0.001*(Target_Orbit['apogee']-self.booster.R0):.0f}_km_" \
                        f"perigee_{0.001*(Target_Orbit['perigee']-self.booster.R0):.0f}km_" \
                        f"inc_{np.rad2deg(Target_Orbit['i']):.1f}deg_" \
                        f"PL_{max(0,payload_mass_predefined):.0f}kg"
            


        except:
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

                    filename = f"LV_{RecoveryStrategy}_apogee_{0.001*(Target_Orbit['apogee']-self.booster.R0):.0f}_km_" \
                        f"perigee_{0.001*(Target_Orbit['perigee']-self.booster.R0):.0f}km_" \
                        f"inc_{np.rad2deg(Target_Orbit['i']):.1f}deg_" \
                        f"PL_{max(0,payload_mass_predefined):.0f}kg_Failed"
                    
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
        self.Solution['LaunchAz'] = LaunchAz_sol
        if RecoveryStrategy != 'EXP':
            self.Solution['x4_0'] = x4_0_sol
            self.Solution['dt4_reentry'] = dt4_reentry_sol
            self.Solution['dt4_ballistic'] = dt4_ballistic_sol
            self.Solution['dt4_before_landing'] = dt4_before_landing_sol
            self.Solution['dt4_landing'] = dt4_landing_scaled_sol
            self.Solution['x4_before_reentry'] = x4_before_reentry_sol
            self.Solution['x4_after_reentryburn'] = x4_after_reentry_sol
            self.Solution['u4_reentry'] = u4_reentry_scaled_sol
            self.Solution['x4_before_landing'] = x4_before_landing_sol
            self.Solution['x4_landing'] = x4_landing_sol
            self.Solution['u4_landing'] = u4_landing_sol
            if RecoveryStrategy == 'RTLS':
                self.Solution['dt4_boostback'] = dt4_boostback_sol
                self.Solution['x4_boostback'] = x4_boostback_sol
                self.Solution['u4_boostback'] = u4_boostback_sol

        with open('.//Results//'+filename, 'wb') as f:
            pickle.dump(self.Solution, f)


        # detailed solution:
        t1_vec = np.linspace(init_time, init_time+self.booster.N*dt1_sol, self.booster.N+1) + 1.15
        t2_vec = np.linspace(t1_vec[-1], t1_vec[-1]+self.eci.N[0]*dt2_sol, self.eci.N[0]+1)
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
            Qdyn1_sol[k] = 0.5 * (x1_sol[2,k]**2 + x1_sol[3,k]**2) * self.atmosphere.rho_fun(x1_sol[1,k])
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
            gama1_local_sol[k] = np.atan2(x1_sol[3,k], x1_sol[2,k])

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
                    Qdyn4_boostback_sol[k] = 0.5 * self.atmosphere.rho_fun(alt4_boostback_sol[k]) * vel4_boostback_sol[k]**2
            
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
                Qdyn4_ballistic_sol[k] = 0.5 * self.atmosphere.rho_fun(alt4_ballistic_sol[k]) * vel4_ballistic_sol[k]**2

            x4_ballistic_M50_sol = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1_M50(self.rocket_return.scale_x(x4_0_sol), 0, dt4_ballistic_sol/self.rocket_return.scaleT)).full().flatten()
            # before reentry burn phase
            alt4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[0], self.rocket_return.R0+x4_before_reentry_sol[1])) - self.rocket_return.R0
            vel4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[2], x4_before_reentry_sol[3]))
            Qdyn4_before_reentry_sol = 0.5 * self.atmosphere.rho_fun(alt4_before_reentry_sol) * vel4_before_reentry_sol**2

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
                Qdyn4_reentry_sol[k] = 0.5 * self.atmosphere.rho_fun(alt4_reentry_sol[k]) * vel4_reentry_sol[k]**2
                hflux4_reentry_sol[k] = self.rocket_return.Booster_k_empirical*np.sqrt(self.atmosphere.rho_fun(alt4_reentry_sol[k])) * vel4_reentry_sol[k]**3

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
                Qdyn4_before_landing_sol[k] = 0.5 * self.atmosphere.rho_fun(alt4_before_landing_sol[k]) * vel4_before_landing_sol[k]**2
                hflux4_before_landing_sol[k] = self.rocket_return.Booster_k_empirical * np.sqrt(self.atmosphere.rho_fun(alt4_before_landing_sol[k])) * vel4_before_landing_sol[k]**3

            # landing burn phase
            N4_landing = self.rocket_return.N
            alt4_landing_sol, vel4_landing_sol = np.zeros((N4_landing+1,1)), np.zeros((N4_landing+1,1))
            Qdyn4_landing_sol, acc4_landing_sol = np.zeros((N4_landing+1,1)), np.zeros((N4_landing+1,1))
            t4_landing_vec = t4_before_landing_vec[-1] + np.linspace(0, dt4_landing_sol*N4_landing, N4_landing+1).reshape(-1,1)
            for k in range(N4_landing+1):
                acc4_landing_sol[k] = self.rocket_return.specific_acc_fun(self.rocket_return.scale_x(x4_landing_sol[:,k]), u4_landing_scaled_sol[min(k,N4_landing-1)])
                alt4_landing_sol[k] = self.rocket_return.local_to_alt(x4_landing_sol[:,k])
                vel4_landing_sol[k] = ca.norm_2(ca.vertcat(x4_landing_sol[2,k], x4_landing_sol[3,k]))
                Qdyn4_landing_sol[k] = 0.5 * self.atmosphere.rho_fun(alt4_landing_sol[k]) * vel4_landing_sol[k]**2

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
        self.Solution['apogee2_sol'] = i2_sol
        self.Solution['perigee2_sol'] = i2_sol
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

        return self.Solution