#!/usr/bin/env python3

from dataclasses import dataclass, field
import numpy as np
import scipy as sp
import casadi as ca
import Atmosphere_Type as Atm_Type
import LV_Type

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

    rocket_booster: LV_Type.BoosterLaunchVehicle_2D = field(default = None)
    rocket_boostback: LV_Type.BoostBackBurn_2D = field(default = None)
    rocket_eci: LV_Type.LaunchVehicle_ECI = field(default = None)
    rocket_return: LV_Type.BoosterReturn_2D = field(default = None)

    InitialGuess: dict = field(default = None)
    
    dm_RTLS: float = field(default = 28000.0)
    dm_ASDS: float = field(default = 13000.0)

@dataclass
class LV_Optimization(VLType):
    def __init__(self, Target_Orbit: dict, atmosphere: Atm_Type.AtmosphereType):
        super().__init__()

        self.Target_Orbit = Target_Orbit

        # Initialize the rocket models
        self.rocket_booster = LV_Type.BoosterLaunchVehicle_2D(atmosphere)
        self.rocket_boostback = LV_Type.BoostBackBurn_2D()
        self.rocket_eci = LV_Type.LaunchVehicle_ECI()
        self.rocket_return = LV_Type.BoosterReturn_2D(atmosphere)

    def FirstStage_Optimization(self, opti: ca.Opti, dm: float = 0.0):
        x1 = opti.variable(self.rocket_booster.nx, self.N1+1) # [x, z, vx, vz, m]
        u1 = opti.variable(self.rocket_booster.nu, self.N1) # [T_factor, alpha]
        dt1 = opti.variable(1, 1)
        payload_mass = opti.variable(1, 1)

        opti.sym_vars.update({'x1': x1,
                         'u1': u1,
                         'dt1': dt1,
                         'payload_mass': payload_mass})

        # Set the initial state
        init_acc = self.rocket_booster.FirstStage_SL_Thrust / (self.rocket_booster.LV_total_mass + payload_mass) - self.rocket_booster.g0 * (self.rocket_booster.R0 / (self.rocket_booster.R0 + self.rocket_booster.LaunchAltitude))**2
        init_vz = init_acc * self.init_time
        init_vx = 0.0
        init_z = 0.5 * init_acc * self.init_time**2
        init_x = 0.0
        init_m = self.rocket_booster.LV_total_mass + payload_mass - (self.rocket_booster.FirstStage_SL_Thrust / (self.rocket_booster.FirstStage_SL_Isp * self.rocket_booster.g0)) * self.init_time

        opti.subject_to(x1[:,0] == ca.vertcat(init_x, init_z, init_vx, init_vz, init_m))
        opti.subject_to(payload_mass >= 0.0)
        opti.subject_to(dt1*self.N1 >= 120.0)
        # set the constraints:
        for k in range(self.N1):
            # set the dynamics
            opti.subject_to((x1[:,k+1]-self.rocket_booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))/dt1 == 0.0)
            # set the thrust constraints
            opti.subject_to(u1[0,k] <= 1.0)
            opti.subject_to(u1[0,k]/self.rocket_booster.FirstStage_MinThrust_Factor >= 1.0)
            opti.subject_to(u1[1,k]/self.rocket_booster.FirstStage_MaxAlpha <= 1.0)
            opti.subject_to(-u1[1,k]/self.rocket_booster.FirstStage_MaxAlpha <= 1.0)

            ### set the state constraints:
            # Max dynamic pressure
            opti.subject_to(0.5*self.rocket_booster.atmosphere.rho_fun(self.rocket_booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2) <= self.rocket_booster.FirstStage_MaxDynamicPressure)
            # Max aerodynamic loads
            opti.subject_to(u1[1,k]*0.5*self.rocket_booster.atmosphere.rho_fun(self.rocket_booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)<=1000.0)
            opti.subject_to(-u1[1,k]*0.5*self.rocket_booster.atmosphere.rho_fun(self.rocket_booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)<=1000.0)

        ### set the final state constraints
        # Final first stage mass
        opti.subject_to(x1[4,self.N1]/(self.rocket_booster.FirstStage_EmptyMass + payload_mass + dm) >= 1.0)
        # Max dynamic pressure at stage separation
        opti.subject_to(0.5*self.rocket_booster.atmosphere.rho_fun(self.rocket_booster.local_to_alt(x1[:,self.N1]))*(x1[2,self.N1]**2+x1[3,self.N1]**2)<=self.rocket_booster.FirstStage_StageSeparationMaxDynamicPressure)

        self.FirstStage_CalcInitGuess()

        # Set the initial guess
        opti.set_initial(x1, self.InitialGuess['x1'])
        opti.set_initial(u1, self.InitialGuess['u1'])
        opti.set_initial(dt1, self.InitialGuess['dt1'])
        opti.set_initial(payload_mass, self.InitialGuess['payload_mass'])

        return opti

    def SecondStage_Optimization(self, opti: ca.Opti):
        
        x2 = opti.variable(self.rocket_eci.nx, self.N2+1) # [x, y, z, vx, vy, vz, m]
        u2 = opti.variable(self.rocket_eci.nu, self.N2) # [fx, fy, fz]
        dt2 = opti.variable(1, 1)
        LaunchAz = opti.variable(1, 1)

        # Set the initial state - the final state of the boost phase transformed to ECI
        opti.subject_to(dt2*self.N2 >= 10.0)
        eci_pos, eci_vel = self.rocket_eci.local_to_eci_func(ca.vertcat(opti.sym_vars['x1'][0,self.N1], 0.0, opti.sym_vars['x1'][1,self.N1]), ca.vertcat(opti.sym_vars['x1'][2,self.N1], 0.0, opti.sym_vars['x1'][3,self.N1]), LaunchAz)
        opti.subject_to(x2[0:3,0] == eci_pos)
        opti.subject_to(x2[3:6,0] == eci_vel)
        opti.subject_to(x2[-1,0] == self.rocket_eci.SecondStage_FullMass + opti.sym_vars['payload_mass'])
        for k in range(self.N2):
            # set the dynamics
            opti.subject_to((x2[:,k+1]-self.rocket_eci.dynamics_kp1(x2[:,k], u2[:,k], dt2))/dt2 == 0.0)
            # set the thrust constraints
            opti.subject_to(ca.norm_2(u2[:,k])/self.rocket_eci.SecondStage_Thrust == 1.0)
            # set the time step constraints

        # set the final state constraints
        opti.subject_to(self.rocket_eci.eci_to_alt(x2[:,self.N2])/self.rocket_eci.FairingSeparationAltitude >= 1.0)

        ## phase 3 - Second Stage, after fairing separation
        x3 = opti.variable(self.rocket_eci.nx, self.N3+1) # [x, y, z, vx, vy, vz, m]
        u3 = opti.variable(self.rocket_eci.nu, self.N3)
        dt3 = opti.variable(1, 1)
        # Set the initial state - the final state of the boost phase transformed to ECI
        opti.subject_to(dt3/0.5 >= 1.0)
        opti.subject_to(x3[0:6,0] == x2[0:6,self.N2])
        opti.subject_to(x3[6,0]/(x2[6,self.N2] - self.rocket_eci.FairingMass) == 1.0)
        for k in range(self.N3):
            # set the dynamics
            # opti.subject_to(x3[:,k+1] == rocket_eci.dynamics_kp1(x3[:,k], u3[:,k], dt3*(1-0.5*k/(N3-1))))
            opti.subject_to((x3[:,k+1]-self.rocket_eci.dynamics_kp1(x3[:,k], u3[:,k], dt3))/dt3 == 0.0)
            # set the thrust constraints
            opti.subject_to(ca.norm_2(u3[:,k])/self.rocket_eci.SecondStage_Thrust == 1.0)
            # set the time step constraints

        # set the final state constraints (Parking orbit)
        h = ca.cross(x3[0:3,self.N3], x3[3:6,self.N3])
        cos_i = h[2] / ca.norm_2(h)
        i = ca.acos(ca.fmin(ca.fmax(cos_i, -1.0), 1.0))
        # i = ca.acos(h[2] / ca.norm_2(h))
        e = ca.cross(x3[3:6,self.N3], h) / (self.rocket_eci.mu) - x3[0:3,self.N3] / ca.norm_2(x3[0:3,self.N3])
        eps = 0.5*ca.sumsqr(x3[3:6,self.N3]) - self.rocket_eci.mu / ca.norm_2(x3[0:3,self.N3])
        a = -self.rocket_eci.mu / (2.0*eps)
        apogee = a * (1.0 + ca.norm_2(e))
        perigee = a * (1.0 - ca.norm_2(e))

        # The optimzed orbit is a parking orbit where:
        # 1. The apogee is the same as the target orbit perigee
        # 2. The perigee is at least 100 km above the earth surface
        # 3. Need to calculate the propellent mass needed to reach the target orbit - raise the current perigee to the target orbit apogee
        # 3a. This manuever is done in the current apogee (which is the same as the target orbit perigee)
        v_apogee = ca.sqrt(self.rocket_eci.mu * (2.0 / self.Target_Orbit["perigee"] - 1.0 / a))
        v_desired = ca.sqrt(self.rocket_eci.mu * (2.0 / self.Target_Orbit["perigee"] - 1.0 / self.Target_Orbit["a"]))
        propellent_mass_for_final_dv = x3[6,self.N3] * (1.0 - ca.exp(-ca.fabs(v_apogee - v_desired) / (self.rocket_eci.SecondStage_Vac_Isp * self.rocket_eci.g0)))

        # set the final state constraints
        opti.subject_to(x3[6,self.N3]/(self.rocket_eci.SecondStage_EmptyMass + opti.sym_vars['payload_mass'] + propellent_mass_for_final_dv) >= 1.0)
        opti.subject_to(apogee/self.Target_Orbit["perigee"] == 1.0)
        opti.subject_to(perigee / (self.rocket_eci.R0 + self.rocket_eci.ParkingOrbit_PerigeeAlt) >= 1.0)
        # opti.subject_to(i == self.Target_Orbit["i"])
        opti.subject_to(i <= self.Target_Orbit["i"] + np.deg2rad(0.1))
        opti.subject_to(i >= self.Target_Orbit["i"] - np.deg2rad(0.1))

        # Save symbolic variables
        opti.sym_vars.update({'x2': x2, 'u2': u2, 'dt2': dt2, 
                              'x3': x3, 'u3': u3, 'dt3': dt3, 'LaunchAz': LaunchAz})

        self.SecondStage_CalcInitialGuess()

        # Set the initial guess
        opti.set_initial(LaunchAz, self.InitialGuess['LaunchAz'])
        opti.set_initial(x2, self.InitialGuess['x2'])
        opti.set_initial(u2, self.InitialGuess['u2'])
        opti.set_initial(dt2, self.InitialGuess['dt2'])
        opti.set_initial(x3, self.InitialGuess['x3'])
        opti.set_initial(u3, self.InitialGuess['u3'])
        opti.set_initial(dt3, self.InitialGuess['dt3'])

        return opti
    

    def BoosterReturn_Optimization(self, opti: ca.Opti, RecoveryStrategy: str, x4_init: np.ndarray = None):

        x4_0 = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_before_reentry = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_after_reentry = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
        u4_reentry = opti.variable(self.rocket_return.nu, 1) # [T_factor]
        x4_before_landing = opti.variable(self.rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_landing = opti.variable(self.rocket_return.nx, self.N4+1) # [x, z, vx, vz, m]
        u4_landing = opti.variable(self.rocket_return.nu, self.N4) # [T_factor]
        dt4_reentry = opti.variable(1)
        dt4_ballistic = opti.variable(1)
        dt4_before_landing = opti.variable(1)
        dt4_landing = opti.variable(1)
        if RecoveryStrategy == 'RTLS':
            dt4_boostback = opti.variable(1)
            x4_boostback = opti.variable(self.rocket_boostback.nx, 1) # [x, z, vx, vz, m]
            u4_boostback = opti.variable(self.rocket_boostback.nu, 1) # [Tx, Tz]

        # set the time step constraints
        if RecoveryStrategy == 'RTLS':
            opti.subject_to(dt4_boostback >= 1.0)
        opti.subject_to(dt4_ballistic >= 1.0)
        opti.subject_to(dt4_reentry >= 1.0)
        opti.subject_to(dt4_before_landing >= 1.0)
        opti.subject_to(dt4_landing*self.N4 >= 1.0)

        # dynamics
        if x4_init is None:
            opti.subject_to(x4_0[0:4,0] == opti.sym_vars['x1'][0:4,self.N1])
            opti.subject_to(x4_0[4,0] == opti.sym_vars['x1'][4,self.N1] - self.rocket_eci.SecondStage_FullMass - opti.sym_vars['payload_mass'])
        else:
            opti.subject_to(x4_0 == x4_init)
        if RecoveryStrategy == 'RTLS':
            opti.subject_to(x4_boostback == self.rocket_boostback.dynamics_kp1_M20(x4_0, u4_boostback, dt4_boostback))
            opti.subject_to(ca.norm_2(u4_boostback) <= 3.0/9.0)
            # opti.subject_to(x4_boostback[2] <= 0.0)

            opti.subject_to(x4_before_reentry == self.rocket_return.dynamics_kp1_M50(x4_boostback, 0, dt4_ballistic))

        else:
            opti.subject_to((x4_before_reentry-self.rocket_return.dynamics_kp1_M50(x4_0, 0, dt4_ballistic)) == 0.0)

        opti.subject_to(self.rocket_return.local_to_alt(x4_before_reentry)/60000.0 == 1.0)
        # Dynamics of entry burn
        opti.subject_to((x4_after_reentry - self.rocket_return.dynamics_kp1_M20(x4_before_reentry, u4_reentry, dt4_reentry)) == 0.0)
        # set the thrust constraints
        opti.subject_to(u4_reentry*3.0 <= 1.0)
        opti.subject_to(u4_reentry*3.0 >= self.rocket_return.FirstStage_MinThrust_Factor)
        opti.subject_to(ca.norm_2(ca.vertcat(x4_after_reentry[2], x4_after_reentry[3])) <= 1400.0)
        opti.subject_to(self.rocket_return.local_to_alt(x4_after_reentry)/45000.0 == 1.0)

        # After Reentry, before landing burn
        opti.subject_to(x4_before_landing == self.rocket_return.dynamics_kp1_M20(x4_after_reentry, 0, dt4_before_landing))
        # opti.subject_to(rocket_return.local_to_alt(x4_before_landing)/1000.0 == 1.0)
        
        # N_before_landing_constraints = 5
        # x4_before_landing_intren = x4_after_reentry
        # for i in range(N_before_landing_constraints):
        #     x4_before_landing_intren = self.rocket_return.dynamics_kp1(x4_before_landing_intren, 0, dt4_before_landing/N_before_landing_constraints)
        #     opti.subject_to(0.5 * self.rocket_return.atmosphere.rho_fun(self.rocket_return.local_to_alt(x4_before_landing_intren)) * ca.norm_2(ca.vertcat(x4_before_landing_intren[2], x4_before_landing_intren[3]))**2 <= self.rocket_return.Booster_MaxDynamicPressure)
            # opti.subject_to(0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren)) * ca.norm_2(ca.vertcat(x4_before_landing_intren[2], x4_before_landing_intren[3]))**3 <= rocket_return.Booster_MaxHeatFlux)

        # Landing Burn:
        opti.subject_to(x4_landing[:,0] == x4_before_landing)
        for k in range(self.N4):
            opti.subject_to((x4_landing[:,k+1]-self.rocket_return.dynamics_kp1(x4_landing[:,k], u4_landing[:,k], dt4_landing)) == 0.0)
                
            opti.subject_to(u4_landing[k] <= 1.0/9.0)
            opti.subject_to(u4_landing[k] >= 0.9*self.rocket_return.EmptyFirstStageMass*self.rocket_return.g0/self.rocket_return.FirstStage_SL_Thrust)


        if RecoveryStrategy == 'RTLS':
            opti.subject_to(x4_landing[0,self.N4] == 0.0)
            opti.subject_to(x4_landing[1,self.N4] == 0.0)
        else:
            # zero altitude at the end of the flight
            opti.subject_to(self.rocket_return.local_to_alt(x4_landing[:,self.N4]) == 0.0) # landing altitude
        # minimal velocity at the end of the flight
        opti.subject_to(ca.norm_2(x4_landing[2:4,self.N4]) == 1.0) # zero velocity at the end of the flight

        if x4_init is not None:
            cost = 0
            cost += -x4_landing[4,self.N4]
            cost += ca.sumsqr(u4_landing[1:] - u4_landing[:-1]) # minimize the thrust changes
            # cost += ca.sumsqr(u4_boostback[0,1:] - u4_boostback[0,:-1]) # minimize the thrust changes
            # cost += ca.sumsqr(u4_boostback[1,1:] - u4_boostback[1,:-1]) # minimize the thrust changes
            opti.minimize(cost)

        # Set the initial guess
        if x4_init is None:
            x4_init = np.zeros((self.rocket_return.nx,))
            x4_init[0:4] = self.InitialGuess['x1'][0:4,self.N1]
            x4_init[4] = self.InitialGuess['x1'][4, self.N1] - self.rocket_eci.SecondStage_FullMass - self.InitialGuess['payload_mass']

        self.BoosterReturn_CalcInitialGuess(RecoveryStrategy = RecoveryStrategy, x4_init = x4_init)

        opti.set_initial(x4_0, self.InitialGuess['x4_0'])
        opti.set_initial(x4_before_reentry, self.InitialGuess['x4_before_reentry'])
        opti.set_initial(x4_after_reentry, self.InitialGuess['x4_after_reentry'])
        opti.set_initial(u4_reentry, self.InitialGuess['u4_reentry'])
        opti.set_initial(x4_before_landing, self.InitialGuess['x4_before_landing'])
        opti.set_initial(x4_landing, self.InitialGuess['x4_landing'])
        opti.set_initial(u4_landing, self.InitialGuess['u4_landing'])
        opti.set_initial(dt4_reentry, self.InitialGuess['dt4_reentry'])
        opti.set_initial(dt4_ballistic, self.InitialGuess['dt4_ballistic'])
        opti.set_initial(dt4_before_landing, self.InitialGuess['dt4_before_landing'])
        opti.set_initial(dt4_landing, self.InitialGuess['dt4_landing'])
        if RecoveryStrategy == 'RTLS':
            opti.set_initial(dt4_boostback, self.InitialGuess['dt4_boostback'])
            opti.set_initial(x4_boostback, self.InitialGuess['x4_boostback'])
            opti.set_initial(u4_boostback, self.InitialGuess['u4_boostback'])

        opti.sym_vars.update({'x4_0': x4_0, 'x4_before_reentry': x4_before_reentry,
                              'x4_after_reentry': x4_after_reentry, 'u4_reentry': u4_reentry,
                              'x4_before_landing': x4_before_landing, 'x4_landing': x4_landing,
                              'u4_landing': u4_landing, 'dt4_reentry': dt4_reentry,
                              'dt4_ballistic': dt4_ballistic, 'dt4_before_landing': dt4_before_landing,
                              'dt4_landing': dt4_landing})
        if RecoveryStrategy == 'RTLS':
            opti.sym_vars.update({'dt4_boostback': dt4_boostback, 'x4_boostback': x4_boostback,
                                  'u4_boostback': u4_boostback})
            
        return opti


    def Define_Cost(self, opti: ca.Opti):
        
        x3 = opti.sym_vars['x3']
        h = ca.cross(x3[0:3,self.N3], x3[3:6,self.N3])
        cos_i = h[2] / ca.norm_2(h)
        i = ca.acos(ca.fmin(ca.fmax(cos_i, -1.0), 1.0))

        cost = 0.0
        # cost = 0.5*(ca.sumsqr(u1[0,:]) + ca.sumsqr(u1[1,:])/rocket_Booster.FirstStage_MaxAlpha**2)
        cost += 0.5*10*(ca.sumsqr(opti.sym_vars['u1'][0,1:]-opti.sym_vars['u1'][0,:-1]) + ca.sumsqr(opti.sym_vars['u1'][1,1:]-opti.sym_vars['u1'][1,:-1]) )*10
        cost += -0.5*opti.sym_vars['payload_mass']
        cost += ca.sumsqr((i-self.Target_Orbit["i"])/np.deg2rad(0.2))
        # if RecoveryStrategy != 'None':
        #     cost += ca.sumsqr(u4_landing[1:] - u4_landing[:-1])/N4 # minimize the thrust changes
        #     cost += ca.sumsqr(u4_reentry)
        opti.minimize(cost)

        return opti

    def FirstStage_CalcInitGuess(self):
        # calculate the initial guess for first stage 
        x1_guess = np.zeros((self.rocket_booster.nx, self.N1+1))
        u1_guess = np.zeros((self.rocket_booster.nu, self.N1))
        
        
        if self.RecoveryStrategy == 'RTLS':
            dm = self.dm_RTLS
            vf1_guess = 6000.0 / 3.6
            dt1_guess = 135.0 / self.N1
        elif self.RecoveryStrategy == 'ASDS':
            dm = self.dm_ASDS
            vf1_guess = 8000.0 / 3.6
            dt1_guess = 140.0 / self.N1
        else:
            dt1_guess = 145.0 / self.N1
            dm = 0.0
            vf1_guess = 8500.0 / 3.6

        v_perigee  =  ca.sqrt(self.rocket_booster.mu * (2.0 / self.Target_Orbit["perigee"] - 1.0 / self.Target_Orbit["a"]))
        factor = np.exp((v_perigee - vf1_guess) / (self.rocket_booster.SecondStage_Vac_Isp * self.rocket_booster.g0))
        payload_mass_guess = 0.75*(self.rocket_booster.SecondStage_FullMass - factor * self.rocket_booster.SecondStage_EmptyMass)/(factor - 1.0)

        init_acc_guess = self.rocket_booster.FirstStage_SL_Thrust / (self.rocket_booster.LV_total_mass + payload_mass_guess) - self.rocket_booster.g0 * (self.rocket_booster.R0 / (self.rocket_booster.R0 + self.rocket_booster.LaunchAltitude))**2
        init_vz_guess = init_acc_guess * self.init_time
        init_vx_guess = 0.0
        init_z_guess = 0.5 * init_acc_guess * self.init_time**2
        init_x_guess = 0.0
        init_m_guess = self.rocket_booster.LV_total_mass + payload_mass_guess - (self.rocket_booster.FirstStage_SL_Thrust / (self.rocket_booster.FirstStage_SL_Isp * self.rocket_booster.g0)) * self.init_time
        x0_guess = np.array([init_x_guess, init_z_guess, init_vx_guess, init_vz_guess, init_m_guess])

        x1_guess[:,0] = x0_guess
        iter=0

        while iter < 20:
            for k in range(self.N1):
                if x1_guess[1,k]<4500 or x1_guess[1,k]>9000:
                    u1_guess[0,k] = 1.0
                else:
                    u1_guess[0,k] = 0.75
                if np.atan2(x1_guess[3,k], x1_guess[2,k]) > 88.0*np.pi/180.0 and np.linalg.norm(x1_guess[2:4,k]) > 75.0:
                    u1_guess[1,k] = -np.deg2rad(1.0)
                else:
                    u1_guess[1,k] = 0.0
                
                alt = self.rocket_booster.local_to_alt(x1_guess[:,k])
                Q = 0.5 * (x1_guess[2,k]**2 + x1_guess[3,k]**2) * self.rocket_booster.atmosphere.rho_fun(alt)
                x1_guess[:,k+1] = self.rocket_booster.dynamics_kp1(x1_guess[:,k], u1_guess[:,k], dt1_guess).full().flatten()
            if 0.5 * self.rocket_booster.atmosphere.rho_fun(self.rocket_booster.local_to_alt(x1_guess[:,self.N1])) * np.linalg.norm(x1_guess[2:4,self.N1]) > self.rocket_booster.FirstStage_StageSeparationMaxDynamicPressure:
                dt1_guess += 1.0 / self.N1
                iter += 1
            elif x1_guess[4,self.N1] <= self.rocket_booster.FirstStage_EmptyMass + payload_mass_guess + dm:
                dt1_guess -= 1.0 / self.N1
                iter += 1
            else:
                break

        self.InitialGuess = {'x1': x1_guess,
                             'u1': u1_guess,
                             'dt1': dt1_guess,
                             'payload_mass': payload_mass_guess }

    
    def FirstStage_SetInitialGuess(self, opti: ca.Opti, InitialGuess: dict):
        # Set the initial guess
        opti.set_initial(opti.sym_vars['x1'], InitialGuess['x1'])
        opti.set_initial(opti.sym_vars['u1'], InitialGuess['u1'])
        opti.set_initial(opti.sym_vars['dt1'], InitialGuess['dt1'])
        opti.set_initial(opti.sym_vars['payload_mass'], InitialGuess['payload_mass'])

        self.InitialGuess['x1'] = InitialGuess['x1']
        self.InitialGuess['u1'] = InitialGuess['u1']
        self.InitialGuess['dt1'] = InitialGuess['dt1']
        self.InitialGuess['payload_mass'] = InitialGuess['payload_mass']

        return opti
    


    def SecondStage_CalcInitialGuess(self):

        if self.InitialGuess is None:
            print("First stage initial guess is not set")
            raise ValueError("First stage initial guess is not set")

        LaunchAz_guess = np.pi/2.0 - self.Target_Orbit["i"]
        dt2_guess = 50.0 / self.N2
        x2_guess = np.zeros((self.rocket_eci.nx, self.N2+1))
        u2_guess = np.zeros((self.rocket_eci.nu, self.N2))
        pos, vel = self.rocket_eci.local_to_eci_func(ca.vertcat(self.InitialGuess['x1'][0,self.N1], 0.0, self.InitialGuess['x1'][1,self.N1]), ca.vertcat(self.InitialGuess['x1'][2, self.N1], 0.0, self.InitialGuess['x1'][3,self.N1]), LaunchAz_guess)
        x2_guess[0:3,0] = pos.full().flatten()
        x2_guess[3:6,0] = vel.full().flatten()
        x2_guess[6,0] = self.rocket_eci.SecondStage_FullMass + self.InitialGuess['payload_mass']
        iter=0
        while iter < 10:
            for k in range(self.N2):
                u2_guess[0:3,k] = (x2_guess[3:6,k] / ca.norm_2(x2_guess[3:6,k])).full().flatten() * self.rocket_eci.SecondStage_Thrust
                x2_guess[:,k+1] = self.rocket_eci.dynamics_kp1(x2_guess[:,k], u2_guess[:,k], dt2_guess).full().flatten()
                alt = ca.norm_2(x2_guess[0:3,k+1]) - self.rocket_eci.R0 / np.sqrt(1.0 - self.rocket_eci.e**2 * np.sin(np.deg2rad(self.rocket_eci.LaunchLatitude))**2)
            if abs(alt-self.rocket_eci.FairingSeparationAltitude) < 100.0:
                break
            else:
                alt_N1 = self.rocket_booster.local_to_alt(self.InitialGuess['x1'][:,self.N1]) - self.rocket_booster.R0
                dt2_guess = dt2_guess*(self.rocket_eci.FairingSeparationAltitude-alt_N1)/ (alt-alt_N1)
                iter += 1

    # calculate the initial guess for second stage after fairing separation
        dt3_guess = 370.0/self.N3
        x3_guess = np.zeros((self.rocket_eci.nx, self.N3+1))
        u3_guess = np.zeros((self.rocket_eci.nu, self.N3))
        iter=0
        
        alpha3_init = 0.0
        v3_max = ca.sqrt(self.rocket_eci.mu * (2.0 / self.Target_Orbit["perigee"] - 1.0 / self.Target_Orbit["apogee"]))
        a_parking = 0.5 * (self.Target_Orbit["perigee"] + (self.rocket_eci.R0 + self.rocket_eci.ParkingOrbit_PerigeeAlt))
        v3_apogee_guess = ca.sqrt(self.rocket_eci.mu * (2.0 / self.Target_Orbit["perigee"] - 1.0 / a_parking))
        v3_desired_guess = ca.sqrt(self.rocket_eci.mu * (2.0 / self.Target_Orbit["perigee"] - 1.0 / self.Target_Orbit["a"]))
        propellent_mass_for_final_dv3_guess = (self.rocket_eci.SecondStage_EmptyMass + self.InitialGuess['payload_mass'])* (1.0 - ca.exp((v3_apogee_guess - v3_desired_guess) / (self.rocket_eci.SecondStage_Vac_Isp * self.rocket_eci.g0)))
        x3_guess[:,0] = x2_guess[:,self.N2]
        while iter < 50:
            alpha3, alpha_factor = alpha3_init, 1.0
            for k in range(self.N3):
                alt = np.linalg.norm(x3_guess[0:3,k]) - self.rocket_eci.R0 / np.sqrt(1.0 - self.rocket_eci.e**2 * np.sin(np.deg2rad(self.rocket_eci.LaunchLatitude))**2)
                h = np.cross(x3_guess[0:3,k], x3_guess[3:6,k])
                h = h / np.linalg.norm(h)
                i = np.arccos(h[2] / np.linalg.norm(h))
                u3_guess[0:3,k] = (x3_guess[3:6,k] / np.linalg.norm(x3_guess[3:6,k])) 
                alpha3 = -alpha3_init*alpha_factor
                u3_guess[0:3,k] = u3_guess[0:3,k] * np.cos(alpha3) + np.cross(h, u3_guess[0:3,k]) * np.sin(alpha3) + h * np.dot(h, u3_guess[0:3,k]) * (1 - np.cos(alpha3)) 
                
                u3_guess[0:3,k] *= self.rocket_eci.SecondStage_Thrust
                x3_guess[:,k+1] = self.rocket_eci.dynamics_kp1(x3_guess[:,k], u3_guess[:,k], dt3_guess).full().flatten()
                h = np.cross(x3_guess[0:3,k+1], x3_guess[3:6,k+1])
                e = ca.cross(x3_guess[3:6,k+1], h) / (self.rocket_eci.mu) - x3_guess[0:3,k+1] / ca.norm_2(x3_guess[0:3,k+1])
                eps = 0.5*ca.sumsqr(x3_guess[3:6,k+1]) - self.rocket_eci.mu / ca.norm_2(x3_guess[0:3,k+1])
                a = -self.rocket_eci.mu / (2.0*eps)
                perigee = a * (1.0 - ca.norm_2(e))
                apogee = a * (1.0 + ca.norm_2(e))
                i = ca.acos(h[2] / ca.norm_2(h))
                
                if x3_guess[6,k+1] < self.rocket_eci.SecondStage_EmptyMass + self.InitialGuess['payload_mass'] + propellent_mass_for_final_dv3_guess:
                    break


            if k < self.N3-1:
                iter += 1
                dt3_guess = dt3_guess * (k+1) / self.N3
            elif x3_guess[6,k+1] < self.rocket_eci.SecondStage_EmptyMass + self.InitialGuess['payload_mass']:
                dt3_guess -= 0.25
                iter += 1
            elif apogee < self.rocket_eci.ParkingOrbit_PerigeeAlt + self.rocket_eci.R0 or a < 0.5*(self.Target_Orbit['perigee']+self.rocket_eci.R0+200.0*1e3):
                dt3_guess += 0.1
                iter += 1
            elif perigee < self.rocket_eci.R0+10*10**3:
                alpha3_init -= np.deg2rad(0.5)
                # dt3_guess += 0.
                iter += 1
            elif apogee > self.Target_Orbit['perigee'] + 150.0*1e3:
                dt3_guess -= 0.1
                iter += 1
            else:
                break
            
        self.InitialGuess['x2'] = x2_guess
        self.InitialGuess['u2'] = u2_guess
        self.InitialGuess['dt2'] = dt2_guess
        self.InitialGuess['x3'] = x3_guess
        self.InitialGuess['u3'] = u3_guess
        self.InitialGuess['dt3'] = dt3_guess
        self.InitialGuess['LaunchAz'] = LaunchAz_guess

        return

    def SecondStage_SetInitialGuess(self, opti: ca.Opti, InitialGuess: dict):
        # Set the initial guess
        opti.set_initial(opti.sym_vars['LaunchAz'], InitialGuess['LaunchAz'])
        opti.set_initial(opti.sym_vars['x2'], InitialGuess['x2'])
        opti.set_initial(opti.sym_vars['u2'], InitialGuess['u2'])
        opti.set_initial(opti.sym_vars['dt2'], InitialGuess['dt2'])
        opti.set_initial(opti.sym_vars['x3'], InitialGuess['x3'])
        opti.set_initial(opti.sym_vars['u3'], InitialGuess['u3'])
        opti.set_initial(opti.sym_vars['dt3'], InitialGuess['dt3'])

        self.InitialGuess['LaunchAz'] = InitialGuess['LaunchAz']
        self.InitialGuess['x2'] = InitialGuess['x2']
        self.InitialGuess['u2'] = InitialGuess['u2']
        self.InitialGuess['dt2'] = InitialGuess['dt2']
        self.InitialGuess['x3'] = InitialGuess['x3']
        self.InitialGuess['u3'] = InitialGuess['u3']
        self.InitialGuess['dt3'] = InitialGuess['dt3']

        return opti
    
    def BoosterReturn_CalcInitialGuess(self, RecoveryStrategy: str, x4_init: np.ndarray):

        # calculate the initial guess for booster return
        x4_0_guess = np.zeros((self.rocket_return.nx, 1))
        x4_0_guess = x4_init
        if RecoveryStrategy == 'RTLS':
            dt4_boostback_guess = 30.0
            x4_boostback_guess = np.zeros((self.rocket_boostback.nx, 1))
            u4_boostback_guess = np.zeros((self.rocket_boostback.nu, 1))
            iter = 0
            while iter < 1:
                u4_boostback_guess[0] = -3.0/9.0 * np.cos(np.deg2rad(-10.0))
                u4_boostback_guess[1] = -3.0/9.0 * np.sin(np.deg2rad(-10.0))
                x4_boostback_guess = self.rocket_boostback.dynamics_kp1_M20(x4_0_guess, u4_boostback_guess, dt4_boostback_guess).full().flatten()
                alt = self.rocket_Booster.local_to_alt(x4_boostback_guess)
                vel = np.linalg.norm(x4_boostback_guess[2:4])
                t_bal = (-x4_boostback_guess[3]  + np.sqrt(x4_boostback_guess[3]**2 +2*self.rocket_boostback.g0*alt))/(0.5*self.rocket_boostback.g0)
                x_bal = x4_boostback_guess[0] + x4_boostback_guess[2]*t_bal
                if abs(x_bal) > 100.0:
                    dt4_boostback_guess += np.clip(-0.01*x_bal / x4_boostback_guess[2], -1.0, 1.0)
                    iter += 1
                else:
                    break

        # calculate the initial guess for booster return
        dt4_ballistic_guess = 200
        iter=0
        while iter < 50:
            if RecoveryStrategy == 'RTLS':
                x4_before_reentry_guess = self.rocket_return.dynamics_kp1_M50(x4_boostback_guess, 0, dt4_ballistic_guess)
            else:
                x4_before_reentry_guess = self.rocket_return.dynamics_kp1_M50(x4_0_guess, 0, dt4_ballistic_guess)
            alt = ca.norm_2(ca.vertcat(x4_before_reentry_guess[0,0], self.rocket_return.R0+x4_before_reentry_guess[1,0])) - self.rocket_return.R0
            vel = ca.norm_2(ca.vertcat(x4_before_reentry_guess[2,0], x4_before_reentry_guess[3,0]))
            if abs(alt - 60000) > 10:
                dt4_ballistic_guess += np.clip((alt - 60000)/abs(x4_before_reentry_guess[3,0]), -10.0, 10.0)
                iter += 1
            else:
                break
        
        u4_reentry_guess = 0.2
        dt4_reentry_guess = 5.0
        v_final = 1400.0
        iter = 0
        while iter < 50:
            x4_after_reentry_guess = self.rocket_return.dynamics_kp1_M20(x4_before_reentry_guess, u4_reentry_guess, dt4_reentry_guess)
            alt = ca.norm_2(ca.vertcat(x4_after_reentry_guess[0,0], self.rocket_return.R0+x4_after_reentry_guess[1,0])) - self.rocket_return.R0
            vel = ca.norm_2(ca.vertcat(x4_after_reentry_guess[2,0], x4_after_reentry_guess[3,0]))
            mass = x4_after_reentry_guess[4,0]
            if abs(vel- v_final) > 10:
                dt4_reentry_guess += 0.5 * np.sign(vel - v_final)
            else:
                break
            iter += 1

        # after reentry burn before landing burn
        iter = 0
        dt4_before_landing_guess = 50.0
        alt4_before_landing = 1000.0
        while iter < 50:
            x4_before_landing_guess = self.rocket_return.dynamics_kp1_M20(x4_after_reentry_guess, 0, dt4_before_landing_guess)
            alt = ca.norm_2(ca.vertcat(x4_before_landing_guess[0,0], self.rocket_return.R0+x4_before_landing_guess[1,0])) - self.rocket_return.R0
            vel = ca.norm_2(ca.vertcat(x4_before_landing_guess[2,0], x4_before_landing_guess[3,0]))
            if abs(alt - alt4_before_landing) > 10:
                dt4_before_landing_guess += 0.9*(alt - alt4_before_landing)/vel
                iter += 1
            else:
                break

        # initial guess for the landing burn
        vel = np.linalg.norm(x4_before_landing_guess[2:4])
        alt  = self.rocket_return.local_to_alt(x4_before_landing_guess)
        dt4_landing_guess = 0.5*vel**2/alt / self.N4
        x4_landing_guess = np.zeros((self.rocket_return.nx, self.N4+1))
        u4_landing_guess = np.zeros((self.rocket_return.nu, self.N4))
        iter = 0
        thrust_factor = 1.0
        while iter < 20:
            u4_landing_guess[0,0] = 0.1111
            x4_landing_guess[:,0] = x4_before_landing_guess.full().flatten()
            for k in range(self.N4):
                vel = np.linalg.norm(x4_landing_guess[2:4,k])
                alt = self.rocket_return.local_to_alt(x4_landing_guess[:,k])
                # if vel > 75.0:
                #     u4_landing_guess[0,k] = 0.1111 * thrust_factor
                # else:
                #     u4_landing_guess[0,k] = x4_landing_guess[4,k] * rocket_return.g0/rocket_return.FirstStage_SL_Thrust
                g_vec = -np.array([x4_landing_guess[0,k], self.rocket_return.R0 + x4_landing_guess[1,k]])
                g_vec /= np.linalg.norm(g_vec)
                g = self.rocket_return.g0 * (self.rocket_return.R0/(self.rocket_return.R0 + x4_landing_guess[1,k]))**2
                v_g = np.dot(g_vec, x4_landing_guess[2:4,k])
                u4_landing_guess[0,k] = np.dot(g_vec, x4_landing_guess[2:4,k])/vel * x4_landing_guess[4,k] * g/self.rocket_return.FirstStage_SL_Thrust
                if vel > 1.0:
                    u4_landing_guess[0,k] += (v_g**2)/(2*alt)*x4_landing_guess[4,k]/self.rocket_return.FirstStage_SL_Thrust
                x4_landing_guess[:,k+1] = self.rocket_return.dynamics_kp1(x4_landing_guess[:,k], u4_landing_guess[0,k], dt4_landing_guess).full().flatten()
                alt = self.rocket_return.local_to_alt(x4_landing_guess[:,k+1])
                vel = np.linalg.norm(x4_landing_guess[2:4,k+1])
                if alt < -1.0:
                    break
            if alt > 20:
                dt4_landing_guess += 0.2*(alt)/vel/self.N4
                iter += 1
            elif alt < -1.0:
                dt4_landing_guess *= k/self.N4
                iter += 1
            elif abs(vel) > 20:
                thrust_factor += 0.5*vel/10
                iter += 1
            else:
                break

        self.InitialGuess['x4_0'] = x4_0_guess
        self.InitialGuess['x4_before_reentry'] = x4_before_reentry_guess
        self.InitialGuess['x4_after_reentry'] = x4_after_reentry_guess
        self.InitialGuess['x4_before_landing'] = x4_before_landing_guess
        self.InitialGuess['x4_landing'] = x4_landing_guess
        self.InitialGuess['u4_reentry'] = u4_reentry_guess
        self.InitialGuess['u4_landing'] = u4_landing_guess
        self.InitialGuess['dt4_reentry'] = dt4_reentry_guess
        self.InitialGuess['dt4_ballistic'] = dt4_ballistic_guess
        self.InitialGuess['dt4_before_landing'] = dt4_before_landing_guess
        self.InitialGuess['dt4_landing'] = dt4_landing_guess
        if RecoveryStrategy == 'RTLS':
            self.InitialGuess['dt4_boostback'] = dt4_boostback_guess
            self.InitialGuess['x4_boostback'] = x4_boostback_guess
            self.InitialGuess['u4_boostback'] = u4_boostback_guess


        return
        
    def Define_Solver(self, opti: ca.Opti, n_iter: int =1000, tol: float =1e-6, mu_strategy: str = 'adaptive'):
        # set the solver
        opts = {
                "print_time": 0,  # Print timing, 
                "ipopt": {
                "linear_solver": "ma97", "hsllib": "/usr/local/lib/libcoinhsl.so",  # MA97 solver Path to HSL library
                # "tol": tol,  # Convergence tolerance
                "max_iter": n_iter,  # Max iterations
                # "print_level": 0,  # Verbosity level
                # "alpha_for_y": "min",  # Fraction-to-boundary rule parameter
                # "timing_statistics": "yes", # Enable timing statistics
                # "nlp_scaling_method": "equilibration-based", # "none", "gradient-based", "equilibration-based" Scaling method
                # 'nlp_scaling_max_gradient': 100,
                # "obj_scaling_factor": 1e-2, # Scaling factor for the objective function
                "mu_strategy": mu_strategy,  # "monotone" or "adaptive" Strategy for updating the barrier parameter
                # "mu_init": 1e-1,  # 1e-1 Initial value of the barrier parameter
                # "mu_min": 1e-11,  # 1e-11 Smallest mu allowed — optimization stops when mu ≤ this
                # "mu_target": 0.0,  # 0.0 IPOPT drives mu towards this target; if 0, tries to reach exact KKT solution
                # "barrier_tol_factor": 2.0,  # Affects how mu and tolerances interact during convergence
                # "alpha_for_y": "min",  # inf, 1.0 Maximum step size for the line search
            }}
        opti.solver('ipopt', opts)

        return opti
    
    def Build_Optimization(self, n_iter: int =1000, tol: float =1e-6, RecoveryStrategy: str = 'None'):
        # Create the optimization problem
        self.RecoveryStrategy = RecoveryStrategy

        opti = ca.Opti()
        opti.sym_vars = {}
        opti = self.FirstStage_Optimization(opti)
        opti = self.SecondStage_Optimization(opti)
        opti = self.Define_Cost(opti)
        opti = self.Define_Solver(opti, n_iter, tol, mu_strategy='monotone')
        if self.RecoveryStrategy != 'None':
            opti = self.BoosterReturn_Optimization(opti, self.RecoveryStrategy)

        return opti

    def Calc_WarmStart(self, n_iter: int =1000, tol: float =1e-6, RecoveryStrategy: str = 'None'):
        # Create the optimization problem
        self.RecoveryStrategy = RecoveryStrategy
        dm = 0.0
        if self.RecoveryStrategy == 'RTLS':
            dm = self.dm_RTLS
        elif self.RecoveryStrategy == 'ASDS':
            dm = self.dm_ASDS
        opti_Launch = ca.Opti()
        opti_Launch.sym_vars = {}
        opti_Launch = self.FirstStage_Optimization(opti_Launch, dm=dm)
        opti_Launch = self.SecondStage_Optimization(opti_Launch)
        opti_Launch = self.Define_Cost(opti_Launch)
        opti_Launch = self.Define_Solver(opti_Launch, n_iter, tol)
        sol_Launch = opti_Launch.solve()

        WarmStart = {}
        WarmStart['x1'] = sol_Launch.value(opti_Launch.sym_vars['x1'])
        WarmStart['u1'] = sol_Launch.value(opti_Launch.sym_vars['u1'])
        WarmStart['dt1'] = sol_Launch.value(opti_Launch.sym_vars['dt1'])
        WarmStart['payload_mass'] = sol_Launch.value(opti_Launch.sym_vars['payload_mass'])
        WarmStart['LaunchAz'] = sol_Launch.value(opti_Launch.sym_vars['LaunchAz'])
        WarmStart['x2'] = sol_Launch.value(opti_Launch.sym_vars['x2'])
        WarmStart['u2'] = sol_Launch.value(opti_Launch.sym_vars['u2'])
        WarmStart['dt2'] = sol_Launch.value(opti_Launch.sym_vars['dt2'])
        WarmStart['x3'] = sol_Launch.value(opti_Launch.sym_vars['x3'])
        WarmStart['u3'] = sol_Launch.value(opti_Launch.sym_vars['u3'])
        WarmStart['dt3'] = sol_Launch.value(opti_Launch.sym_vars['dt3'])

        if self.RecoveryStrategy != 'None':
            x4_Init = np.zeros((self.rocket_return.nx,))
            x4_Init[0:4] = sol_Launch.value(opti_Launch.sym_vars['x1'])[0:4,self.N1]
            x4_Init[4] = sol_Launch.value(opti_Launch.sym_vars['x1'])[4,self.N1] - self.rocket_eci.SecondStage_FullMass - sol_Launch.value(opti_Launch.sym_vars['payload_mass'])
            opti_Return = ca.Opti()
            opti_Return.sym_vars = {}
            opti_Return = self.BoosterReturn_Optimization(opti_Return, self.RecoveryStrategy, x4_Init)
            opti_Return = self.Define_Solver(opti=opti_Return, n_iter=500)

            sol_Return = opti_Return.solve()

            WarmStart['x4_0'] = sol_Return.value(opti_Return.sym_vars['x4_0'])
            WarmStart['x4_before_reentry'] = sol_Return.value(opti_Return.sym_vars['x4_before_reentry'])
            WarmStart['x4_after_reentry'] = sol_Return.value(opti_Return.sym_vars['x4_after_reentry'])
            WarmStart['x4_before_landing'] = sol_Return.value(opti_Return.sym_vars['x4_before_landing'])
            WarmStart['x4_landing'] = sol_Return.value(opti_Return.sym_vars['x4_landing'])
            WarmStart['u4_reentry'] = sol_Return.value(opti_Return.sym_vars['u4_reentry'])
            WarmStart['u4_landing'] = sol_Return.value(opti_Return.sym_vars['u4_landing'])
            WarmStart['dt4_reentry'] = sol_Return.value(opti_Return.sym_vars['dt4_reentry'])
            WarmStart['dt4_ballistic'] = sol_Return.value(opti_Return.sym_vars['dt4_ballistic'])
            WarmStart['dt4_before_landing'] = sol_Return.value(opti_Return.sym_vars['dt4_before_landing'])
            WarmStart['dt4_landing'] = sol_Return.value(opti_Return.sym_vars['dt4_landing'])
            if RecoveryStrategy == 'RTLS':
                WarmStart['dt4_boostback'] = sol_Return.value(opti_Return.sym_vars['dt4_boostback'])
                WarmStart['x4_boostback'] = sol_Return.value(opti_Return.sym_vars['x4_boostback'])
                WarmStart['u4_boostback'] = sol_Return.value(opti_Return.sym_vars['u4_boostback'])
            
        return WarmStart
    
    def Set_WarmStart(self, opti: ca.Opti, WarmStart: dict):
        # Set the initial guess
        opti.set_initial(opti.sym_vars['x1'], WarmStart['x1'])
        opti.set_initial(opti.sym_vars['u1'], WarmStart['u1'])
        opti.set_initial(opti.sym_vars['dt1'], WarmStart['dt1'])
        opti.set_initial(opti.sym_vars['payload_mass'], WarmStart['payload_mass'])
        opti.set_initial(opti.sym_vars['LaunchAz'], WarmStart['LaunchAz'])
        opti.set_initial(opti.sym_vars['x2'], WarmStart['x2'])
        opti.set_initial(opti.sym_vars['u2'], WarmStart['u2'])
        opti.set_initial(opti.sym_vars['dt2'], WarmStart['dt2'])
        opti.set_initial(opti.sym_vars['x3'], WarmStart['x3'])
        opti.set_initial(opti.sym_vars['u3'], WarmStart['u3'])
        opti.set_initial(opti.sym_vars['dt3'], WarmStart['dt3'])

        if self.RecoveryStrategy != 'None':
            opti.set_initial(opti.sym_vars['x4_0'], WarmStart['x4_0'])
            opti.set_initial(opti.sym_vars['x4_before_reentry'], WarmStart['x4_before_reentry'])
            opti.set_initial(opti.sym_vars['x4_after_reentry'], WarmStart['x4_after_reentry'])
            opti.set_initial(opti.sym_vars['x4_before_landing'], WarmStart['x4_before_landing'])
            opti.set_initial(opti.sym_vars['x4_landing'], WarmStart['x4_landing'])
            opti.set_initial(opti.sym_vars['u4_reentry'], WarmStart['u4_reentry'])
            opti.set_initial(opti.sym_vars['u4_landing'], WarmStart['u4_landing'])
            opti.set_initial(opti.sym_vars['dt4_reentry'], WarmStart['dt4_reentry'])
            opti.set_initial(opti.sym_vars['dt4_ballistic'], WarmStart['dt4_ballistic'])
            opti.set_initial(opti.sym_vars['dt4_before_landing'], WarmStart['dt4_before_landing'])
            opti.set_initial(opti.sym_vars['dt4_landing'], WarmStart['dt4_landing'])

            if self.RecoveryStrategy == 'RTLS':
                opti.set_initial(opti.sym_vars['dt4_boostback'], WarmStart['dt4_boostback'])
                opti.set_initial(opti.sym_vars['x4_boostback'], WarmStart['x4_boostback'])
                opti.set_initial(opti.sym_vars['u4_boostback'], WarmStart['u4_boostback'])
        return opti