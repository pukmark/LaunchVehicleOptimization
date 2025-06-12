#!/usr/bin/env pythorocket_eci.N[1]

import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.gridspec as gridspec
import casadi as ca
import pickle
import json


import Atmosphere_Type as Atm_Type
import LV_Type_scaled as LV_Type
import LV_Plot as LVp

if __name__ == '__main__':
    RecoveryStrategy = 'RTLS' # 'None', 'RTLS', 'ASDS'
    payload_mass_predefined = -1 # payload mass defined in kg, maximmize the payload mass if set to non-positive value
    optimize_stage_partition = False
    Plot_interm = True

    # Create an instance of the AtmosphereType class
    atmosphere = Atm_Type.AtmosphereType()
    # Create an instance of the VLType class
    booster = LV_Type.BoosterLaunchVehicle_2D(atmosphere=atmosphere, N=60)
    rocket_eci = LV_Type.LaunchVehicle_ECI(N = [10, 60])
    rocket_boostback = LV_Type.BoostBackBurn_2D() if RecoveryStrategy == 'RTLS' else None
    rocket_return = LV_Type.BoosterReturn_2D(atmosphere=atmosphere, N=15) if RecoveryStrategy != 'None' else None
    LV_plot = LVp.LV_plot(booster, rocket_eci, rocket_boostback, rocket_return)

    # Target orbit parameters for LEO (ISS)
    # Target_Orbit = {"apogee": booster.R0 + 420.0*1000.0, # semi-major axis
    #                 "perigee": booster.R0 + 400.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(53.3), # inclination [rad]
    #                 }

    # Target orbit parameters for MEO
    # Target_Orbit = {"apogee": booster.R0 + 20196.0*1e3, # semi-major axis
    #                 "perigee": booster.R0 + 1193.0*1e3, # semi-minor axis
    #                 "i": np.deg2rad(55.0), # inclination [rad]
    #                 }

    # Target orbit parameters for GTO
    Target_Orbit = {"apogee": booster.R0 + 35786.0*1000.0, # semi-major axis
                    "perigee": booster.R0 + 200.0*1000.0, # semi-minor axis
                    "i": np.deg2rad(booster.LaunchLatitude), # inclination [rad]
                    }

    # Target orbit parameters for TLI
    # Target_Orbit = {"apogee": 384400*1000.0, # apogee [m]
    #                 "perigee": booster.R0 + 200.0*1000.0, # perigee [m]
    #                 "i": np.deg2rad(booster.LaunchLatitude), # inclination [rad]
    #                 }

    # Target orbit parameters for SSO
    # Target_Orbit = {"apogee": booster.R0 + 780 * 1000.0, # semi-major axis
    #                 "perigee": booster.R0 + 780.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(98.6), # inclination [rad]
    #                 }

    Target_Orbit["a"] = 0.5 * (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # semi-major axis
    Target_Orbit["e"] = (Target_Orbit["apogee"] - Target_Orbit["perigee"]) / (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # eccentricity

    # define the optimization problem
    opti = ca.Opti()

    # First Phase - Booster 
    if payload_mass_predefined > 0.0:
        payload_mass_scaled = payload_mass_predefined / booster.scaleX[4]
    else:
        payload_mass_scaled = opti.variable(1, 1)
        opti.subject_to(payload_mass_scaled >= 0.0)
    if optimize_stage_partition:
        FirstStage_Propellent_fraction = opti.variable(1, 1)
        opti.subject_to(FirstStage_Propellent_fraction >= 0.0)
        opti.subject_to(FirstStage_Propellent_fraction <= 1.0)
    else:
        FirstStage_Propellent_fraction = booster.FirstStagePropellentMass / booster.TotalPropellentMass
    
    payload_mass = payload_mass_scaled * booster.scaleX[4]
    FirstStage_Propellent = booster.TotalPropellentMass * FirstStage_Propellent_fraction
    SecondStage_Propellent = booster.TotalPropellentMass * (1.0 - FirstStage_Propellent_fraction)
    SecondStage_FullMass = SecondStage_Propellent + rocket_eci.EmptySecondStageMass + rocket_eci.FairingMass

    x1 = opti.variable(booster.nx, booster.N+1) # [x, z, vx, vz, m]
    u1 = opti.variable(booster.nu, booster.N) # [T_factor, alpha]
    dt1 = opti.variable(1)
    LaunchAz = opti.variable(1)

    # Set the initial state
    init_time = 5.0
    init_acc = booster.FirstStage_SL_Thrust / (booster.LV_total_mass + payload_mass) - booster.g0 * (booster.R0 / (booster.R0 + booster.LaunchAltitude))**2
    init_vz = init_acc * init_time
    init_vx = 0.0
    init_z = 0.5 * init_acc * init_time**2
    init_x = 0.0
    init_m = booster.LV_total_mass + payload_mass - (booster.FirstStage_SL_Thrust / (booster.FirstStage_SL_Isp * booster.g0)) * init_time

    opti.subject_to(x1[:,0] == booster.scale_x(ca.vertcat(init_x, init_z, init_vx, init_vz, init_m)))
    opti.subject_to(dt1 >= 0.1)
    # set the constraints:
    for k in range(booster.N):
        # set the dynamics
        opti.subject_to(x1[:,k+1] == booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))
        # set the thrust constraints
        opti.subject_to(u1[0,k] <= 1.0)
        opti.subject_to(u1[0,k] >= booster.FirstStage_MinThrust_Factor)
        opti.subject_to(u1[1,k]**2 <= 1.0)

        # set the state constraints
        uk = booster.unscale_u(u1[:,k])
        q = booster.dynamic_pressure_fun(x1[:,k])

        opti.subject_to(q/booster.FirstStage_MaxDynamicPressure <= 1.0) # dynamic pressure limit
        opti.subject_to((uk[1] * q / 1e4)**2 <= 1.0) # alpha * dynamic pressure limit
        opti.subject_to(booster.specific_acc_fun(x1[:,k], u1[:,k])/booster.Payload_Max_acc <= 1.0)

    # set the final state constraints    
    x1f = booster.unscale_x(x1[:,booster.N])
    # q1f = 0.5 * (x1f[2]**2 + x1f[3]**2) * atmosphere.rho_fun(booster.local_to_alt(x1f))
    opti.subject_to(booster.dynamic_pressure_fun(x1[:,booster.N])/booster.FirstStage_StageSeparationMaxDynamicPressure <= 1.0)
    if RecoveryStrategy == 'None':
            opti.subject_to(x1f[4] >= rocket_eci.EmptyFirstStageMass + SecondStage_FullMass + payload_mass)

    # Phase 2 - Second Stage, before fairing separation (if needed)
    x2 = opti.variable(rocket_eci.nx, rocket_eci.N[0]+1) # [x, y, z, vx, vy, vz, m]
    u2 = opti.variable(rocket_eci.nu, rocket_eci.N[0]) # [fx, fy, fz]
    dt2 = opti.variable(1)

    # Set the initial state - the final state of the boost phase transformed to ECI
    opti.subject_to(dt2 >= 0.1)
    opti.subject_to(LaunchAz >= -np.pi)
    opti.subject_to(LaunchAz <= np.pi)
    eci_pos, eci_vel = rocket_eci.local_to_eci_func(ca.vertcat(x1f[0], 0.0, x1f[1]), ca.vertcat(x1f[2], 0.0, x1f[3]), LaunchAz)
    x2_init = ca.vertcat(eci_pos, eci_vel, SecondStage_FullMass + payload_mass)
    opti.subject_to(x2[:,0] == rocket_eci.scale_x(x2_init))
    for k in range(rocket_eci.N[0]):
        # set the dynamics
        opti.subject_to(x2[:,k+1] == rocket_eci.dynamics_kp1(x2[:,k], u2[:,k], dt2))
        # set the thrust constraints
        opti.subject_to(ca.norm_2(u2[:,k]) <= 1.0)
        opti.subject_to(ca.norm_2(u2[:,k]) >= rocket_eci.SecondStage_MinThrust_Factor)

    # set the final state constraints
    opti.subject_to(rocket_eci.eci_to_alt(rocket_eci.unscale_x(x2[:,rocket_eci.N[0]]))/rocket_eci.FairingSeparationAltitude >= 1.0)

    ## phase 3 - Second Stage, after fairing separation
    x3 = opti.variable(rocket_eci.nx, rocket_eci.N[1]+1) # [x, y, z, vx, vy, vz, m]
    u3 = opti.variable(rocket_eci.nu, rocket_eci.N[1])
    dt3 = opti.variable(1)
    # Set the initial state - the final state of the boost phase transformed to ECI
    x3_init = ca.vertcat(x2[0:6,rocket_eci.N[0]], x2[6,rocket_eci.N[0]] - rocket_eci.FairingMass/ rocket_eci.scaleX[6])
    opti.subject_to(dt3 >= 0.1)
    opti.subject_to(x3[:,0] == x3_init)
    for k in range(rocket_eci.N[1]):
        # set the dynamics
        opti.subject_to(x3[:,k+1] == rocket_eci.dynamics_kp1(x3[:,k], u3[:,k], dt3))
        # set the thrust constraints
        opti.subject_to(ca.norm_2(u3[:,k]) <= 1.0)
        opti.subject_to(ca.norm_2(u3[:,k]) >= rocket_eci.SecondStage_MinThrust_Factor)
        opti.subject_to(rocket_eci.specific_acc_fun(x3[:,k], u3[:,k])/booster.Payload_Max_acc <= 1.0)

    # set the final state constraints (Parking orbit)
    x3f = rocket_eci.unscale_x(x3[:,rocket_eci.N[1]])
    h = ca.cross(x3f[0:3], x3f[3:6])
    i = ca.acos(ca.fmin(ca.fmax(h[2] / ca.norm_2(h), -1.0), 1.0))
    e = ca.cross(x3f[3:6], h) / (rocket_eci.mu) - x3f[0:3] / ca.norm_2(x3f[0:3])
    eps = 0.5*ca.sumsqr(x3f[3:6]) - rocket_eci.mu / ca.norm_2(x3f[0:3])
    a = -rocket_eci.mu / (2.0*eps)
    apogee = a * (1.0 + ca.norm_2(e))
    perigee = a * (1.0 - ca.norm_2(e))

    # The optimzed orbit is a parking orbit where:
    # 1. The apogee is the same as the target orbit perigee
    # 2. The perigee is at least 100 km above the earth surface
    # 3. Need to calculate the propellent mass needed to reach the target orbit - raise the current perigee to the target orbit apogee
    # 3a. This manuever is done in the current apogee (which is the same as the target orbit perigee)
    v_apogee = ca.sqrt(rocket_eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / a))
    v_desired = ca.sqrt(rocket_eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["a"]))
    propellent_mass_for_final_dv = x3f[6] * (1.0 - ca.exp(-ca.fabs(v_apogee - v_desired) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))
    
    # v_final_apogee = ca.sqrt(rocket_eci.mu * (1.0 / Target_Orbit["a"]))
    # dv_for_inclination_fix = 2 * v_apogee * ca.sin(Target_Orbit["i"]/2.0)
    # propellent_mass_for_inclination_fix = (x3[6,rocket_eci.N[1]]-propellent_mass_for_final_dv) * (1.0 - ca.exp(-ca.fabs(dv_for_inclination_fix) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))


    # propellent_mass_for_final_dv = 0.0
    x3f_mass_scaled = (rocket_eci.SecondStage_EmptyMass + payload_mass + propellent_mass_for_final_dv) / rocket_eci.scaleX[6]
    opti.subject_to(x3[6,rocket_eci.N[1]] >= x3f_mass_scaled)
    opti.subject_to(apogee/Target_Orbit["perigee"] == 1.0)
    opti.subject_to(perigee/(rocket_eci.R0 + rocket_eci.ParkingOrbit_PerigeeAlt) >= 1.0)
    opti.subject_to(i/Target_Orbit["i"] == 1.0)


# Phase 4 - Booster Return:
    N4 = 15
    if RecoveryStrategy != 'None':
        if RecoveryStrategy == 'RTLS':
            dt4_boostback = opti.variable(1)
            x4_boostback = opti.variable(rocket_boostback.nx, 1) # [x, z, vx, vz, m]
            u4_boostback = opti.variable(rocket_boostback.nu, 1) # [Tx, Tz]
        x4_0 = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_before_reentry = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_after_reentryburn = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        u4_reentry = opti.variable(rocket_return.nu, 1) # [T_factor]
        x4_before_landing = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_landing = opti.variable(rocket_return.nx, N4+1) # [x, z, vx, vz, m]
        u4_landing = opti.variable(rocket_return.nu, N4) # [T_factor]
        dt4_reentry = opti.variable(1)
        dt4_ballistic = opti.variable(1)
        dt4_before_landing = opti.variable(1)
        dt4_landing = opti.variable(1)
        


        # set the time step constraints
        if RecoveryStrategy == 'RTLS':
            opti.subject_to(dt4_boostback >= 0.1)
        opti.subject_to(dt4_ballistic >= 100.0)
        opti.subject_to(dt4_reentry >= 0.1)
        opti.subject_to(dt4_before_landing >= 0.1)
        opti.subject_to(dt4_landing >= 0.1)

        # Set the initial state
        x4_init_unscaled = ca.vertcat(x1f[0:4], x1f[4] - SecondStage_FullMass - payload_mass)
        
        # Boostback phase and ballistic phase
        if RecoveryStrategy == 'RTLS':
            opti.subject_to(x4_0 == rocket_boostback.scale_x(x4_init_unscaled))
            opti.subject_to(x4_boostback == rocket_boostback.dynamics_kp1_M20(x4_0, u4_boostback, dt4_boostback))
            opti.subject_to(ca.norm_2(u4_boostback) <= 1.0)

            opti.subject_to(x4_before_reentry == rocket_return.dynamics_kp1_M50(rocket_return.scale_x(rocket_boostback.unscale_x(x4_boostback)), 0, dt4_ballistic))
        else:
            opti.subject_to(x4_0 == rocket_return.scale_x(x4_init_unscaled))
            opti.subject_to(x4_before_reentry == rocket_return.dynamics_kp1_M50(x4_0, 0, dt4_ballistic))
        opti.subject_to(rocket_return.local_to_alt(rocket_return.unscale_x(x4_before_reentry))/60000.0 >= 1.0)

        # rentry burn
        opti.subject_to(x4_after_reentryburn  == rocket_return.dynamics_kp1_M10(x4_before_reentry, u4_reentry, dt4_reentry))
        ### set the thrust constraints
        opti.subject_to(u4_reentry[0] <= 1.0)
        opti.subject_to(u4_reentry[0] >= rocket_return.FirstStage_MinThrust_Factor)

        ### After Reentry, before landing burn (ballistic)
        opti.subject_to(x4_before_landing == rocket_return.dynamics_kp1_M20(x4_after_reentryburn, 0, dt4_before_landing))

        N_before_landing_constraints = 10
        x4_before_landing_intren = x4_after_reentryburn
        for i in range(N_before_landing_constraints):
            x4_before_landing_intren = rocket_return.dynamics_kp1(x4_before_landing_intren, 0, dt4_before_landing/N_before_landing_constraints)
            opti.subject_to(rocket_return.dynamic_pressure_fun(x4_before_landing_intren)/rocket_return.Booster_MaxDynamicPressure <= 1.0)
            opti.subject_to(rocket_return.heat_flux_fun(x4_before_landing_intren)/rocket_return.Booster_MaxHeatFlux <= 1.0)

        ### Landing Burn:
        opti.subject_to(x4_landing[:,0] == x4_before_landing)
        for k in range(N4):
            opti.subject_to(x4_landing[:,k+1] == rocket_return.dynamics_kp1(x4_landing[:,k], u4_landing[:,k], dt4_landing))
            opti.subject_to(u4_landing[0,k] <= 1.0/3.0) 
            opti.subject_to(u4_landing[0,k] >= 0.75*1.0/3.0*rocket_return.EmptyFirstStageMass*rocket_return.g0/rocket_return.scaleU[0])

        # final state constraints
        x4f_landing_unscaled = rocket_return.unscale_x(x4_landing[:,N4])
        if RecoveryStrategy == 'RTLS':
            opti.subject_to(ca.norm_2(x4_landing[0:2,N4]) == 0.0)
        else:
            # zero altitude at the end of the flight
            opti.subject_to(rocket_return.local_to_alt(x4f_landing_unscaled) == 0.0) # landing altitude
        # minimal velocity at the end of the flight
        opti.subject_to(ca.norm_2(x4_landing[2:4,N4]) <= 1.0/rocket_return.scaleX[2]) # minimal velocity at the end of the flight
        opti.subject_to(x4_landing[4,N4] >= rocket_return.EmptyFirstStageMass/rocket_return.scaleX[4]) # minimal mass at the end of the flight


    # set the cost function
    cost = 0.0
    cost += 0.5*1e-3*(ca.sumsqr(u1[0,1:]-u1[0,:-1]) + ca.sumsqr(u1[1,1:]-u1[1,:-1]))
    cost += -payload_mass_scaled*100
    if type(payload_mass_scaled) != ca.MX:
        cost += -x3f[6]*10
    opti.minimize(cost)


    # set the solver
    
    opts = {# "print_time": 1,  # Print timing, 
            "ipopt": {
            "linear_solver": "ma97", "hsllib": "/usr/local/lib/libcoinhsl.so",  # MA97 solver Path to HSL library
            "tol": 1e-5,  # Convergence tolerance
            # "max_iter": 500,  # Max iterations
            "print_level": 5,  # Verbosity level
            # "alpha_for_y": "min",  # Fraction-to-boundary rule parameter
            # "timing_statistics": "yes", # Enable timing statistics
            # "nlp_scaling_method": "none", # 'none' 'gradient-based', # Scaling method
            # 'nlp_scaling_max_gradient': 100,
            # "obj_scaling_factor": 1e-4, # Scaling factor for the objective function
            # "mu_strategy": "adaptive",  # "adaptive" or "adaptive" Strategy for updating the barrier parameter
            # "mu_init": 1e-11,  # 1e-1 Initial value of the barrier parameter
            # "mu_min": 1e-11,  # 1e-11 Smallest mu allowed — optimization stops when mu ≤ this
            # "mu_target": 0.0,  # 0.0 IPOPT drives mu towards this target; if 0, tries to reach exact KKT solution
            # "barrier_tol_factor": 5.0,  # Affects how mu and tolerances interact during convergence
            "expect_infeasible_problem": "yes",  # More tolerant to infeasibility
            "required_infeasibility_reduction": 1e-8,  # Less strict, delays restoration trigger
            "soft_resto_pderror_reduction_factor": 0.1,  # Soft restoration less aggressive
            "bound_relax_factor": 1e-8,  # Bound infeasibility relaxation very tight (prevents jumps)
            "bound_push": 1e-8,  # Careful step near bounds
        }}
    if Plot_interm:
        LV_plot.plot_init()
        opti.callback(lambda i: LV_plot.plot_iteration(i, opti.debug.value(opti.debug.x), Target_Orbit, payload_mass_predefined, optimize_stage_partition))
    opti.solver('ipopt', opts)

    # calculate the initial guess for first stage 
    x1_guess = np.zeros((booster.nx, booster.N+1))
    u1_guess = np.zeros((booster.nu, booster.N))
    dt1_guess = (150.0-init_time) / booster.N
    LaunchAz_guess = np.pi/2.0 - Target_Orbit["i"]

    a_parking = 0.5 * (150.0*1e3 + rocket_eci.R0 + Target_Orbit["perigee"])
    v_apogee = ca.sqrt(rocket_eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / a_parking))
    v_desired = ca.sqrt(rocket_eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["a"]))
    v_parking = ca.sqrt(rocket_eci.mu * (2.0 / (150.0*1e3 + rocket_eci.R0) - 1.0 / a_parking))
    

    if RecoveryStrategy == 'RTLS':
        v3_0_guess = 1700.0
    elif RecoveryStrategy == 'ASDS':
        v3_0_guess = 2200.0
    else:
        v3_0_guess = 2800.0

    if payload_mass_predefined <= 0.0:
        alpha = ca.exp(ca.fabs(v_parking-v3_0_guess) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0))
        payload_mass_guess = 0.5*(rocket_eci.SecondStage_FullMass - alpha*rocket_eci.SecondStage_EmptyMass)/(alpha-1.0)
    else:
        payload_mass_guess = payload_mass_predefined*0.5
    m3_final = (rocket_eci.SecondStage_EmptyMass + payload_mass_guess)*(1.0 - ca.exp(-ca.fabs(v_apogee - v_desired) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))

    init_acc_guess = booster.FirstStage_SL_Thrust / (booster.LV_total_mass + payload_mass_guess) - booster.g0 * (booster.R0 / (booster.R0 + booster.LaunchAltitude))**2
    init_vz_guess = init_acc_guess * init_time
    init_vx_guess = 0.0
    init_z_guess = 0.5 * init_acc_guess * init_time**2
    init_x_guess = 0.0
    init_m_guess = booster.LV_total_mass + payload_mass_guess - (booster.FirstStage_SL_Thrust / (booster.FirstStage_SL_Isp * booster.g0)) * init_time
    x0_guess = np.array([init_x_guess, init_z_guess, init_vx_guess, init_vz_guess, init_m_guess])

    if RecoveryStrategy == 'RTLS':
        dm = 30000.0
        # dm = 200000.0
    elif RecoveryStrategy == 'ASDS':
        dm = 15000.0
        # dm = 150000.0
    else:    
        dm = 0.0

    x1_guess[:,0] = booster.scale_x(x0_guess).full().flatten()
    iter=0
    while iter < 30:
        for k in range(booster.N):
            xk = booster.unscale_x(x1_guess[:,k]).full().flatten()
            alt = booster.local_to_alt(xk)
            if xk[1]<5000 or xk[1]>10000:
                u1_guess[0,k] = 1.0
            else:
                u1_guess[0,k] = 0.70

            max_u0 = booster.Payload_Max_acc*xk[4]/booster.FirstStage_Vac_Thrust
            u1_guess[0,k] = np.clip(u1_guess[0,k], booster.FirstStage_MinThrust_Factor, max_u0)

            if np.atan2(xk[3], xk[2]) > 87.0*np.pi/180.0 and np.linalg.norm(xk[2:4]) > 75.0:
                u1_guess[1,k] = -np.deg2rad(1.0) / booster.FirstStage_MaxAlpha
            else:
                u1_guess[1,k] = 0.0
            
            x1_guess[:,k+1] = booster.dynamics_kp1(x1_guess[:,k], u1_guess[:,k], dt1_guess).full().flatten()
            acc = booster.specific_acc_fun(x1_guess[:,k], u1_guess[:,k])

        xN = booster.unscale_x(x1_guess[:,booster.N])
        alt_N1 = booster.local_to_alt(xN)
        Q = 0.5 * (xN[2]**2 + xN[3]**2) * atmosphere.rho_fun(alt_N1)
        if Q > booster.FirstStage_StageSeparationMaxDynamicPressure:
            dt1_guess += 0.5 / booster.N
            iter += 1
        elif xN[4] <= booster.FirstStage_EmptyMass + payload_mass_guess + dm:
            dt1_guess -= 1.0 / booster.N
            iter += 1
        else:
            break

    # calculate the initial guess for second stage before fairing separation, if needed
    dt2_guess = 50.0 / rocket_eci.N[0]
    x2_guess = np.zeros((rocket_eci.nx, rocket_eci.N[0]+1))
    u2_guess = np.zeros((rocket_eci.nu, rocket_eci.N[0]))
    x1f_guess = booster.unscale_x(x1_guess[:,booster.N])
    pos, vel = rocket_eci.local_to_eci_func(ca.vertcat(x1f_guess[0], 0.0, x1f_guess[1]), ca.vertcat(x1f_guess[2], 0.0, x1f_guess[3]), LaunchAz_guess)
    x2_init = ca.vertcat(pos, vel, rocket_eci.SecondStage_FullMass + payload_mass_guess)
    x2_guess[:,0] = rocket_eci.scale_x(x2_init).full().flatten()
    iter=0
    while iter < 10:
        for k in range(rocket_eci.N[0]):
            xk = rocket_eci.unscale_x(x2_guess[:,k])
            alt = rocket_eci.eci_to_alt(xk)
            u2_guess[0:3,k] = (x2_guess[3:6,k] / ca.norm_2(x2_guess[3:6,k])).full().flatten()
            x2_guess[:,k+1] = rocket_eci.dynamics_kp1(x2_guess[:,k], u2_guess[:,k], dt2_guess).full().flatten()
        xN = rocket_eci.unscale_x(x2_guess[:,rocket_eci.N[0]])
        alt = rocket_eci.eci_to_alt(xN)
        if abs(alt-rocket_eci.FairingSeparationAltitude) < 100.0:
            break
        else:
            dt2_guess = dt2_guess*(rocket_eci.FairingSeparationAltitude-alt_N1)/ (alt-alt_N1)
            iter += 1

# calculate the initial guess for second stage after fairing separation
    dt3_guess = 370.0/rocket_eci.N[1]
    x3_guess = np.zeros((rocket_eci.nx, rocket_eci.N[1]+1))
    u3_guess = np.zeros((rocket_eci.nu, rocket_eci.N[1]))
    x3_guess[0:6,0] = x2_guess[0:6,rocket_eci.N[0]]
    x3_guess[6,0] = x2_guess[6,rocket_eci.N[0]] - rocket_eci.FairingMass / rocket_eci.scaleX[6]
    iter=0

    alpha3_init = 0.0
    v3_max = ca.sqrt(rocket_eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["apogee"]))
    while iter < 50:
        alpha3, alpha_factor = alpha3_init, 1.0
        for k in range(rocket_eci.N[1]):
            xk = rocket_eci.unscale_x(x3_guess[:,k]).full().flatten()
            h = np.cross(xk[0:3], xk[3:6])
            h = h / np.linalg.norm(h)
            u3_guess[0:3,k] = (xk[3:6] / np.linalg.norm(xk[3:6])) 
            # alpha_factor = k/rocket_eci.N[1]
            alpha3 = -alpha3_init*alpha_factor
            u3_guess[0:3,k] = u3_guess[0:3,k] * np.cos(alpha3) + np.cross(h, u3_guess[0:3,k]) * np.sin(alpha3) + h * np.dot(h, u3_guess[0:3,k]) * (1 - np.cos(alpha3)) 
            acc = rocket_eci.SecondStage_Thrust / xk[6]
            if acc > rocket_eci.Payload_Max_acc:
                u3_guess[0:3,k] = u3_guess[0:3,k] * rocket_eci.Payload_Max_acc/acc
            x3_guess[:,k+1] = rocket_eci.dynamics_kp1(x3_guess[:,k], u3_guess[:,k], dt3_guess).full().flatten()

            xkp1 = rocket_eci.unscale_x(x3_guess[:,k+1]).full().flatten()
            vel = np.linalg.norm(xkp1[3:6])
            h_guess = np.cross(xkp1[0:3], xkp1[3:6])
            e_guess = ca.cross(xkp1[3:6], h_guess) / (rocket_eci.mu) - xkp1[0:3] / ca.norm_2(xkp1[0:3])
            eps_guess = 0.5*ca.sumsqr(xkp1[3:6]) - rocket_eci.mu / ca.norm_2(xkp1[0:3])
            a_guess = -rocket_eci.mu / (2.0*eps_guess)
            perigee_guess = a_guess * (1.0 - ca.norm_2(e_guess))
            apogee_guess = a_guess * (1.0 + ca.norm_2(e_guess))
            i_guess = ca.acos(h_guess[2] / ca.norm_2(h_guess))


            
            if xkp1[6] < m3_final + rocket_eci.SecondStage_EmptyMass + payload_mass_guess:
                break
        if k < rocket_eci.N[1]-1:
            iter += 1
            dt3_guess = dt3_guess * (k+1) / rocket_eci.N[1]
        elif xkp1[6] < m3_final:
            dt3_guess -= 0.25
            iter += 1
        elif apogee_guess < rocket_eci.ParkingOrbit_PerigeeAlt + rocket_eci.R0 or a_guess < 0.5*(Target_Orbit['perigee']+rocket_eci.R0+200.0*1e3):
            dt3_guess += 0.1
            iter += 1
        elif perigee_guess < rocket_eci.R0+10*10**3:
            alpha3_init -= np.deg2rad(0.5)
            # dt3_guess += 0.
            iter += 1
        elif apogee_guess > Target_Orbit['perigee'] + 150.0*1e3:
            dt3_guess -= 0.1
            iter += 1
        else:
            break


    # calculate the initial guess for booster return
    if RecoveryStrategy != 'None':
        # calculate the initial guess for booster return
        x4_0_guess = np.zeros((rocket_return.nx, 1))
        x1f_guess_unscaled = booster.unscale_x(x1_guess[:,booster.N]).full().flatten()
        x4_0_guess[0:4,0] = x1f_guess_unscaled[0:4]
        x4_0_guess[4,0] = x1f_guess_unscaled[4] - rocket_eci.SecondStage_FullMass - payload_mass_guess
        if RecoveryStrategy == 'RTLS':
            dt4_boostback_guess = 22.0
            x4_boostback_guess = np.zeros((rocket_boostback.nx, 1))
            u4_boostback_guess = np.zeros((rocket_boostback.nu, 1))
            iter = 0
            while iter < 50:
                u4_boostback_guess[0] = -np.cos(np.deg2rad(-10.0))
                u4_boostback_guess[1] = -np.sin(np.deg2rad(-10.0))
                x4_boostback_guess = rocket_boostback.dynamics_kp1_M20(rocket_boostback.scale_x(x4_0_guess), u4_boostback_guess, dt4_boostback_guess).full().flatten()
                x4_boostback_guess_unscaled = rocket_boostback.unscale_x(x4_boostback_guess)
                alt = booster.local_to_alt(x4_boostback_guess_unscaled)
                vel = np.linalg.norm(x4_boostback_guess_unscaled[2:4])
                t_bal = (-x4_boostback_guess_unscaled[3]  + np.sqrt(x4_boostback_guess_unscaled[3]**2 +2*rocket_boostback.g0*alt))/(0.5*rocket_boostback.g0)
                x_bal = x4_boostback_guess_unscaled[0] + x4_boostback_guess_unscaled[2]*400
                if abs(x_bal) > 100.0:
                    dt4_boostback_guess += np.clip(x_bal / abs(x4_boostback_guess_unscaled[2])/400, -1.0, 1.0)
                    iter += 1
                else:
                    break

        # calculate the initial guess for booster return
        dt4_ballistic_guess = 300
        iter=0
        while iter < 50:
            if RecoveryStrategy == 'RTLS':
                x4_before_reentry_guess = rocket_return.dynamics_kp1_M50(rocket_return.scale_x(x4_boostback_guess_unscaled), 0, dt4_ballistic_guess)
            else:
                x4_before_reentry_guess = rocket_return.dynamics_kp1_M50(rocket_return.scale_x(x4_0_guess), 0, dt4_ballistic_guess)
            x4_before_reentry_guess_unscaled = rocket_return.unscale_x(x4_before_reentry_guess)
            alt = rocket_return.local_to_alt(x4_before_reentry_guess_unscaled)
            vel = ca.norm_2(x4_before_reentry_guess_unscaled[2:4,0])
            if abs(alt - 60000) > 10:
                dt4_ballistic_guess += np.clip((alt - 60000)/abs(x4_before_reentry_guess_unscaled[3,0]), -10.0, 10.0)
                iter += 1
            else:
                break
        
        u4_reentry_guess = 1.0
        dt4_reentry_guess = 1.0
        iter = 0
        v_final = 1300.0
        while iter < 50:
            x4_after_reentry_guess = rocket_return.dynamics_kp1_M20(x4_before_reentry_guess, u4_reentry_guess, dt4_reentry_guess)
            x4_after_reentry_guess_unscaled = rocket_return.unscale_x(x4_after_reentry_guess)
            alt = rocket_return.local_to_alt(x4_after_reentry_guess_unscaled)
            vel = ca.norm_2(x4_after_reentry_guess_unscaled[2:4,0])
            mass = x4_after_reentry_guess_unscaled[4,0]
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
            x4_before_landing_guess = rocket_return.dynamics_kp1_M20(x4_after_reentry_guess, 0, dt4_before_landing_guess)
            x4_before_landing_guess_unscaled = rocket_return.unscale_x(x4_before_landing_guess).full().flatten()
            alt = rocket_return.local_to_alt(x4_before_landing_guess_unscaled)
            vel = np.linalg.norm(x4_before_landing_guess_unscaled[2:4])
            if abs(alt - alt4_before_landing) > 10:
                dt4_before_landing_guess += 0.9*(alt - alt4_before_landing)/vel
                iter += 1
            else:
                break

        # initial guess for the landing burn
        vel = np.linalg.norm(x4_before_landing_guess_unscaled[2:4])
        alt  = rocket_return.local_to_alt(x4_before_landing_guess_unscaled)
        dt4_landing_guess = vel/(rocket_return.scaleU[0]/x4_before_landing_guess_unscaled[4]/3-rocket_return.g0) / N4
        x4_landing_guess = np.zeros((rocket_return.nx, N4+1))
        u4_landing_guess = np.zeros((rocket_return.nu, N4))
        iter = 0
        thrust_factor = 1.0
        while iter < 20:
            x4_landing_guess[:,0] = x4_before_landing_guess.full().flatten()
            for k in range(N4):
                xk = rocket_return.unscale_x(x4_landing_guess[:,k]).full().flatten()
                vel = np.linalg.norm(xk[2:4])
                alt = rocket_return.local_to_alt(xk)
                g_vec = -np.array([xk[0], rocket_return.R0 + xk[1]])
                g_vec /= np.linalg.norm(g_vec)
                g = rocket_return.g0 * (rocket_return.R0/(rocket_return.R0 + xk[1]))**2
                g_v = (g_vec[0]*xk[2] + g_vec[1]*xk[3])/vel * g
                u4_landing_guess[0,k] = g_v * xk[4]/rocket_return.scaleU[0]                    
                if vel > 30.0 and alt > 50.0:
                    u4_landing_guess[0,k] += (3*vel**2)/(2*alt)*xk[4]/rocket_return.scaleU[0]
                if alt < 50.0:
                    u4_landing_guess[0,k] += 0.5*(vel-0.5)**2/alt*xk[4]/rocket_return.scaleU[0]
                u4_landing_guess[0,k] = thrust_factor*np.clip(u4_landing_guess[0,k], 0.01, 1.0/3.0)
                x4_landing_guess[:,k+1] = rocket_return.dynamics_kp1(x4_landing_guess[:,k], u4_landing_guess[:,k], dt4_landing_guess).full().flatten()
                xkp1 = rocket_return.unscale_x(x4_landing_guess[:,k+1]).full().flatten()
                alt = rocket_return.local_to_alt(xkp1)
                vel = np.linalg.norm(xkp1[2:4])
                if alt < -1.0:
                    break
            if alt > 10:
                dt4_landing_guess += (alt)/vel/N4
                # thrust_factor *= 1.01
                iter += 1
            # elif vel > 20:
            #     thrust_factor += rocket_return.FirstStage_EmptyMass/rocket_return.FirstStage_SL_Thrust
            #     iter += 1
            elif alt < -1.0 and k<N4-1:
                dt4_landing_guess *= k/(N4-1)
                iter += 1
            elif alt < -1.0:
                thrust_factor *= 1.1
                iter += 1
            else:
                break


    # set the initial guess for the optimization variables
    if type(payload_mass_scaled) == ca.MX:
        opti.set_initial(payload_mass_scaled, payload_mass_guess/booster.scaleX[4])
    if (FirstStage_Propellent_fraction) == ca.MX:
        opti.set_initial(FirstStage_Propellent_fraction, booster.FirstStagePropellentMass/booster.TotalPropellentMass)
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
    if RecoveryStrategy != 'None':
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
            opti.set_initial(x4_0, rocket_boostback.scale_x(x4_0_guess))
            opti.set_initial(dt4_boostback, dt4_boostback_guess)
            opti.set_initial(x4_boostback, x4_boostback_guess)
            opti.set_initial(u4_boostback, u4_boostback_guess)
        else:
            opti.set_initial(x4_0, rocket_return.scale_x(x4_0_guess))


    #  load Solution.pickle
    if RecoveryStrategy == 'RTLS':
        with open('Solution_RTLS.pickle', 'rb') as f:
            Solution = pickle.load(f)
    elif 0:
        with open('Solution1.pickle', 'rb') as f:
            Solution = pickle.load(f)

        if type(payload_mass_scaled) == ca.MX:
            opti.set_initial(payload_mass_scaled, Solution['payload_mass_scaled'])
        opti.set_initial(dt1, Solution['dt1'])
        opti.set_initial(x1, Solution['x1'])
        opti.set_initial(u1, Solution['u1'])
        opti.set_initial(LaunchAz, Solution['LaunchAz'])
        opti.set_initial(dt2, Solution['dt2'])
        opti.set_initial(x2, Solution['x2'])
        opti.set_initial(u2, Solution['u2'])
        opti.set_initial(dt3, Solution['dt3'])
        opti.set_initial(x3, Solution['x3'])
        opti.set_initial(u3, Solution['u3'])
        if RecoveryStrategy != 'None':
            opti.set_initial(dt4_reentry, Solution['dt4_reentry'])
            opti.set_initial(dt4_ballistic, Solution['dt4_ballistic'])
            opti.set_initial(dt4_before_landing, Solution['dt4_before_landing'])
            opti.set_initial(dt4_landing, Solution['dt4_landing'])
            opti.set_initial(x4_before_reentry, Solution['x4_before_reentry'])
            opti.set_initial(x4_after_reentryburn, Solution['x4_after_reentryburn'])
            opti.set_initial(u4_reentry, Solution['u4_reentry'])
            opti.set_initial(x4_before_landing, Solution['x4_before_landing'])
            opti.set_initial(x4_landing, Solution['x4_landing'])
            opti.set_initial(u4_landing, Solution['u4_landing'])
            if RecoveryStrategy == 'RTLS':
                opti.set_initial(x4_0, Solution['x4_0'])
                opti.set_initial(dt4_boostback, Solution['dt4_boostback'])
                opti.set_initial(x4_boostback, Solution['x4_boostback'])
                opti.set_initial(u4_boostback, Solution['u4_boostback'])
            else:
                opti.set_initial(x4_0, Solution['x4_0'])




    # Solve the optimization problem
    if RecoveryStrategy != 'None':
        dt4_landing_scaled_sol = dt4_landing_guess
        x4_landing_scaled_sol  = x4_landing_guess
        u4_landing_scaled_sol  = u4_landing_guess
    
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
        if optimize_stage_partition:
            FirstStage_Propellent_fraction_sol = np.array(sol.value(FirstStage_Propellent_fraction))
        LaunchAz_sol = np.array(sol.value(LaunchAz))
        propellent_mass_for_final_dv_sol = np.array(sol.value(propellent_mass_for_final_dv))
        if RecoveryStrategy != 'None':
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

        # save the solution
        Solution = {}
        Solution['u1'] = u1_scaled_sol
        Solution['x1'] = x1_scaled_sol
        Solution['dt1'] = dt1_scaled_sol
        Solution['u2'] = u2_scaled_sol
        Solution['x2'] = x2_scaled_sol
        Solution['dt2'] = dt2_scaled_sol
        Solution['u3'] = u3_scaled_sol
        Solution['x3'] = x3_scaled_sol
        Solution['dt3'] = dt3_scaled_sol
        if type(payload_mass_scaled) == ca.MX:
            Solution['payload_mass_scaled'] = payload_mass_scaled_sol
        if optimize_stage_partition:
            Solution['FirstStage_Propellent_fraction'] = FirstStage_Propellent_fraction_sol
        Solution['LaunchAz'] = LaunchAz_sol
        if RecoveryStrategy != 'None':
            Solution['dt4_reentry'] = dt4_reentry_scaled_sol
            Solution['dt4_ballistic'] = dt4_ballistic_scaled_sol
            Solution['dt4_before_landing'] = dt4_before_landing_scaled_sol
            Solution['dt4_landing'] = dt4_landing_scaled_sol
            Solution['x4_before_reentry'] = x4_before_reentry_scaled_sol
            Solution['x4_after_reentryburn'] = x4_after_reentry_scaled_sol
            Solution['u4_reentry'] = u4_reentry_scaled_sol
            Solution['x4_before_landing'] = x4_before_landing_scaled_sol
            Solution['x4_landing'] = x4_landing_scaled_sol
            Solution['u4_landing'] = u4_landing_scaled_sol
            if RecoveryStrategy == 'RTLS':
                Solution['dt4_boostback'] = dt4_boostback_scaled_sol
                Solution['x4_boostback'] = x4_boostback_scaled_sol
                Solution['u4_boostback'] = u4_boostback_scaled_sol
        Solution['x4_0'] = x4_0_scaled_sol

        with open('Solution1.pickle', 'wb') as f:
            pickle.dump(Solution, f)

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
        if RecoveryStrategy != 'None':
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
        payload_mass_sol = payload_mass_scaled_sol * booster.scaleX[4]
    else:
        payload_mass_scaled_sol = payload_mass
        payload_mass_sol = payload_mass_scaled_sol
    dt1_sol = dt1_scaled_sol
    dt2_sol = dt2_scaled_sol
    dt3_sol = dt3_scaled_sol
    u1_sol = np.zeros_like(u1_scaled_sol)
    for i in range(u1_scaled_sol.shape[1]):
        u1_sol[:,i] = booster.unscale_u(u1_scaled_sol[:,i]).full().flatten()
    x1_sol = np.zeros_like(x1_scaled_sol)
    for i in range(x1_scaled_sol.shape[1]):
        x1_sol[:,i] = booster.unscale_x(x1_scaled_sol[:,i]).full().flatten()
    u2_sol = np.zeros_like(u2_scaled_sol)
    for i in range(u2_scaled_sol.shape[1]):
        u2_sol[:,i] = rocket_eci.unscale_u(u2_scaled_sol[:,i]).full().flatten()
    x2_sol = np.zeros_like(x2_scaled_sol)
    for i in range(x2_scaled_sol.shape[1]):
        x2_sol[:,i] = rocket_eci.unscale_x(x2_scaled_sol[:,i]).full().flatten()
    x3_sol = np.zeros_like(x3_scaled_sol)
    for i in range(x3_scaled_sol.shape[1]):
        x3_sol[:,i] = rocket_eci.unscale_x(x3_scaled_sol[:,i]).full().flatten()
    u3_sol = np.zeros_like(u3_scaled_sol)
    for i in range(u3_scaled_sol.shape[1]):
        u3_sol[:,i] = rocket_eci.unscale_u(u3_scaled_sol[:,i]).full().flatten()
    if RecoveryStrategy != 'None':
        dt4_reentry_sol = dt4_reentry_scaled_sol
        dt4_ballistic_sol = dt4_ballistic_scaled_sol
        dt4_before_landing_sol = dt4_before_landing_scaled_sol
        dt4_landing_sol = dt4_landing_scaled_sol
        x4_before_reentry_sol = rocket_return.unscale_x(x4_before_reentry_scaled_sol).full().flatten()
        x4_after_reentry_sol = rocket_return.unscale_x(x4_after_reentry_scaled_sol).full().flatten()
        u4_reentry_sol = rocket_return.unscale_u([u4_reentry_scaled_sol]).full().flatten()
        x4_before_landing_sol = rocket_return.unscale_x(x4_before_landing_scaled_sol).full().flatten()
        x4_landing_sol  = np.zeros_like(x4_landing_scaled_sol)
        for i in range(x4_landing_scaled_sol.shape[1]):
            x4_landing_sol[:,i] = rocket_return.unscale_x(x4_landing_scaled_sol[:,i]).full().flatten()
        u4_landing_sol = np.zeros_like(u4_landing_scaled_sol)
        for i in range(u4_landing_scaled_sol.shape[0]):
            u4_landing_sol[i] = rocket_return.unscale_u([u4_landing_scaled_sol[i]]).full().flatten()[0]
        if RecoveryStrategy == 'RTLS':
            x4_0_sol = rocket_boostback.unscale_x(x4_0_scaled_sol).full().flatten()
            dt4_boostback_sol = dt4_boostback_scaled_sol
            x4_boostback_sol = rocket_boostback.unscale_x(x4_boostback_scaled_sol).full().flatten()
            u4_boostback_sol = rocket_boostback.unscale_u(u4_boostback_scaled_sol).full().flatten()
        else:
            x4_0_sol = rocket_return.unscale_x(x4_0_scaled_sol).full().flatten()


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

    plt.ioff()
    t1_vec = np.linspace(init_time, init_time+booster.N*dt1_sol, booster.N+1) + 1.15
    t2_vec = np.linspace(t1_vec[-1], t1_vec[-1]+rocket_eci.N[0]*dt2_sol, rocket_eci.N[0]+1)
    t3_vec = np.linspace(t2_vec[-1], t2_vec[-1]+rocket_eci.N[1]*dt3_sol, rocket_eci.N[1]+1)

    Isp1_sol, acc1_sol = np.zeros((booster.N)), np.zeros((booster.N))
    Qdyn1_sol = np.zeros((booster.N+1))
    a1_sol, b1_sol = np.zeros((booster.N+1)), np.zeros((booster.N+1))
    apogee1_sol, perigee1_sol = np.zeros((booster.N+1)), np.zeros((booster.N+1))
    alt1_sol, vel1_sol = np.zeros((booster.N+1)), np.zeros((booster.N+1))
    i1_sol, gama1_sol, gama1_local_sol = np.zeros((booster.N+1)), np.zeros((booster.N+1)), np.zeros((booster.N+1))
    x1_eci_sol = np.zeros((rocket_eci.nx, booster.N+1))
    for k in range(booster.N+1):
        if k < booster.N:
            Isp1_sol[k] = booster.ISP_calc(x1_sol[:,k], u1_sol[:,k]).full().flatten()[0]
            acc1_sol[k] = booster.specific_acc_fun(x1_scaled_sol[:,k], u1_scaled_sol[:,k])
        Qdyn1_sol[k] = 0.5 * (x1_sol[2,k]**2 + x1_sol[3,k]**2) * atmosphere.rho_fun(x1_sol[1,k])
        pos, vel = rocket_eci.local_to_eci_func(ca.vertcat(x1_sol[0,k], 0.0, x1_sol[1,k]), ca.vertcat(x1_sol[2,k], 0.0, x1_sol[3,k]), LaunchAz_sol)
        x1_eci_sol[0:3,k], x1_eci_sol[3:6,k], x1_eci_sol[6,k] = pos.T, vel.T, x1_sol[4,k]
        alt1_sol[k] = np.linalg.norm(pos) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
        vel1_sol[k] = np.linalg.norm(vel)
        h1 = ca.cross(pos, vel)
        e1 = ca.cross(vel, h1) / (rocket_eci.mu) - pos / np.linalg.norm(pos)
        eps1 = 0.5*ca.sumsqr(vel) - rocket_eci.mu / np.linalg.norm(pos)
        a1_sol[k] = -rocket_eci.mu / (2.0*eps1)
        b1_sol[k] = a1_sol[k] * np.sqrt(1.0 - ca.sumsqr(e1))
        apogee1_sol[k] = a1_sol[k] * (1.0 + np.linalg.norm(e1))
        perigee1_sol[k] = a1_sol[k] * (1.0 - np.linalg.norm(e1))
        i1_sol[k] = np.arccos(h1[2] / ca.norm_2(h1))
        gama1_sol[k] = np.arccos(ca.dot(vel, pos) / (ca.norm_2(vel) * ca.norm_2(pos)))
        gama1_local_sol[k] = np.atan2(x1_sol[3,k], x1_sol[2,k])

    Isp2_sol, acc2_sol = np.zeros((rocket_eci.N[0])), np.zeros((rocket_eci.N[0]))
    Alpha2_sol = np.zeros((rocket_eci.N[0]))
    a2_sol, b2_sol = np.zeros((rocket_eci.N[0]+1)), np.zeros((rocket_eci.N[0]+1))
    apogee2_sol, perigee2_sol = np.zeros((rocket_eci.N[0]+1)), np.zeros((rocket_eci.N[0]+1))
    alt2_sol, i2_sol = np.zeros((rocket_eci.N[0]+1)), np.zeros((rocket_eci.N[0]+1))
    Qdyn2_sol, gama2_sol = np.zeros((rocket_eci.N[0]+1)), np.zeros((rocket_eci.N[0]+1))
    for k in range(rocket_eci.N[0]+1):
        if k < rocket_eci.N[0]:
            Alpha2_sol[k] = np.arccos(np.dot(u2_sol[:,k], x2_sol[3:6,k]) / (rocket_eci.SecondStage_Thrust * ca.norm_2(x2_sol[3:6,k])))
            Isp2_sol[k] = rocket_eci.ISP_calc(x2_sol[:,k], u2_sol[:,k]).full().flatten()[0]
            acc2_sol[k] = rocket_eci.specific_acc_fun(x2_scaled_sol[:,k], u2_scaled_sol[:,k])
        pos, vel = x2_sol[0:3,k], x2_sol[3:6,k]
        alt2_sol[k] = np.linalg.norm(pos) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
        Qdyn2_sol[k] = 0.5 * (x2_sol[3,k]**2 + x2_sol[4,k]**2 + x2_sol[5,k]**2) * atmosphere.rho_fun(alt2_sol[k])
        h2 = ca.cross(pos, vel)
        e2 = ca.cross(vel, h2) / (rocket_eci.mu) - pos / np.linalg.norm(pos)
        eps2 = 0.5*ca.sumsqr(vel) - rocket_eci.mu / np.linalg.norm(pos)
        a2_sol[k] = -rocket_eci.mu / (2.0*eps2)
        b2_sol[k] = a2_sol[k] * np.sqrt(1.0 - ca.sumsqr(e2))
        apogee2_sol[k] = a2_sol[k] * (1.0 + np.linalg.norm(e2))
        perigee2_sol[k] = a2_sol[k] * (1.0 - np.linalg.norm(e2))
        i2_sol[k] = np.arccos(h2[2] / ca.norm_2(h2))
        gama2_sol[k] = np.arccos(np.dot(vel, pos) / (ca.norm_2(vel) * ca.norm_2(pos)))

    Isp3_sol, acc3_sol = np.zeros((rocket_eci.N[1])), np.zeros((rocket_eci.N[1]))
    Alpha3_sol = np.zeros((rocket_eci.N[1]))
    a3_sol, b3_sol = np.zeros((rocket_eci.N[1]+1)), np.zeros((rocket_eci.N[1]+1))
    apogee3_sol, perigee3_sol = np.zeros((rocket_eci.N[1]+1)), np.zeros((rocket_eci.N[1]+1))
    alt3_sol, i3_sol, gama3_sol = np.zeros((rocket_eci.N[1]+1)), np.zeros((rocket_eci.N[1]+1)), np.zeros((rocket_eci.N[1]+1))
    for k in range(rocket_eci.N[1]+1):
        if k < rocket_eci.N[1]:
            Alpha3_sol[k] = np.arccos(np.dot(u3_sol[:,k], x3_sol[3:6,k]) / (rocket_eci.SecondStage_Thrust * ca.norm_2(x3_sol[3:6,k])))
            Isp3_sol[k] = rocket_eci.ISP_calc(x3_sol[:,k], u3_sol[:,k]).full().flatten()[0]
            acc3_sol[k] = rocket_eci.specific_acc_fun(x3_scaled_sol[:,k], u3_scaled_sol[:,k])
        pos, vel = x3_sol[0:3,k], x3_sol[3:6,k]
        alt3_sol[k] = np.linalg.norm(pos) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
        h3 = ca.cross(pos, vel)
        e3 = ca.cross(vel, h3) / (rocket_eci.mu) - pos / np.linalg.norm(pos)
        eps3 = 0.5*ca.sumsqr(vel) - rocket_eci.mu / np.linalg.norm(pos)
        a3_sol[k] = -rocket_eci.mu / (2.0*eps3)
        b3_sol[k] = a3_sol[k] * np.sqrt(1.0 - ca.sumsqr(e3))
        apogee3_sol[k] = a3_sol[k] * (1.0 + np.linalg.norm(e3))
        perigee3_sol[k] = a3_sol[k] * (1.0 - np.linalg.norm(e3))
        i3_sol[k] = np.arccos(h3[2] / ca.norm_2(h3))
        gama3_sol[k] = np.arccos(np.dot(vel, pos) / (ca.norm_2(vel) * ca.norm_2(pos)))

    v3_apogee = ca.sqrt(rocket_eci.mu * (2.0 / apogee3_sol[-1] - 1.0 / a3_sol[-1]))
    v3_desired = ca.sqrt(rocket_eci.mu * (2.0 / apogee3_sol[-1] - 1.0 / Target_Orbit["a"]))
    propellent_mass_for_final_dv3 = x3_sol[6,rocket_eci.N[1]] * (1.0 - ca.exp((v3_apogee - v3_desired) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))
    actual_apogee3 = Target_Orbit["a"] * (1.0 + Target_Orbit["e"])

    if RecoveryStrategy != 'None':
        alt4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[0], rocket_return.R0+x4_0_sol[1])) - rocket_return.R0
        vel4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[2], x4_0_sol[3]))
        
        if RecoveryStrategy == 'RTLS':
            t4_boostback_vec = t1_vec[-1]+np.linspace(0, dt4_boostback_sol, N4+1)
            alt4_boostback_sol = np.zeros((N4+1,1))
            vel4_boostback_sol = np.zeros((N4+1,1))
            x4_boostback_vec_sol = np.zeros((rocket_boostback.nx, N4+1))
            x4_boostback_vec_sol[:,0] = x4_0_sol
            for k in range(N4+1):
                if k<N4:
                    x4_boostback_vec_sol[:,k+1] = rocket_boostback.unscale_x(rocket_boostback.dynamics_kp1(rocket_boostback.scale_x(x4_boostback_vec_sol[:,k]), u4_boostback_scaled_sol, dt4_boostback_sol/N4)).full().flatten()
                alt4_boostback_sol[k] = ca.norm_2(ca.vertcat(x4_boostback_vec_sol[0,k], rocket_return.R0+x4_boostback_vec_sol[1,k])) - rocket_return.R0
                vel4_boostback_sol[k] = ca.norm_2(ca.vertcat(x4_boostback_vec_sol[2,k], x4_boostback_vec_sol[3,k]))
            
        N4_ballistic = 50
        x4_ballistic_sol = np.zeros((rocket_return.nx, N4_ballistic+1))
        alt4_ballistic_sol, vel4_ballistic_sol = np.zeros((N4_ballistic+1,1)), np.zeros((N4_ballistic+1,1))
        Qdyn4_ballistic_sol = np.zeros((N4_ballistic+1,1))
        t4_ballistic_vec = np.zeros((N4_ballistic+1,1))
        if RecoveryStrategy == 'RTLS':
            x4_ballistic_sol[:,0] = rocket_boostback.unscale_x(x4_boostback_scaled_sol).full().flatten()
        else:
            x4_ballistic_sol[:,0] = x4_0_sol
        t4_ballistic_vec[0]  = t1_vec[-1] if RecoveryStrategy != 'RTLS' else dt4_boostback_sol+t1_vec[-1]
        alt4_ballistic_sol[0] = ca.norm_2(ca.vertcat(x4_ballistic_sol[0,0], rocket_return.R0+x4_ballistic_sol[1,0])) - rocket_return.R0
        vel4_ballistic_sol[0] = ca.norm_2(ca.vertcat(x4_ballistic_sol[2,0], x4_ballistic_sol[3,0]))
        Qdyn4_ballistic_sol[0] = 0.5 * atmosphere.rho_fun(alt4_ballistic_sol[0]) * vel4_ballistic_sol[0]**2
        for k in range(N4_ballistic):
            t4_ballistic_vec[k+1] = t4_ballistic_vec[k] + dt4_ballistic_sol/N4_ballistic
            x4_ballistic_sol[:,k+1] = rocket_return.unscale_x(rocket_return.dynamics_kp1(rocket_return.scale_x(x4_ballistic_sol[:,k]), 0, dt4_ballistic_sol/N4_ballistic)).full().flatten()
            alt4_ballistic_sol[k+1] = ca.norm_2(ca.vertcat(x4_ballistic_sol[0,k+1], rocket_return.R0+x4_ballistic_sol[1,k+1])) - rocket_return.R0
            vel4_ballistic_sol[k+1] = ca.norm_2(ca.vertcat(x4_ballistic_sol[2,k+1], x4_ballistic_sol[3,k+1]))
            Qdyn4_ballistic_sol[k+1] = 0.5 * atmosphere.rho_fun(alt4_ballistic_sol[k+1]) * vel4_ballistic_sol[k+1]**2

        x4_ballistic_M50_sol = rocket_return.unscale_x(rocket_return.dynamics_kp1_M50(rocket_return.scale_x(x4_0_sol), 0, dt4_ballistic_sol)).full().flatten()
        # before reentry burn phase
        alt4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[0], rocket_return.R0+x4_before_reentry_sol[1])) - rocket_return.R0
        vel4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[2], x4_before_reentry_sol[3]))
        Qdyn4_before_reentry_sol = 0.5 * atmosphere.rho_fun(alt4_before_reentry_sol) * vel4_before_reentry_sol**2

        # reentry phase
        k_empirical = 1.83e-4 # [sqrt(kg)*s^3/m]
        N4_reentry = 50
        x4_reentry_sol = np.zeros((rocket_return.nx, N4_reentry+1))
        alt4_reentry_sol, vel4_reentry_sol = np.zeros((N4_reentry+1,1)), np.zeros((N4_reentry+1,1))
        Qdyn4_reentry_sol, hflux4_reentry_sol = np.zeros((N4_reentry+1,1)), np.zeros((N4_reentry+1,1))
        t4_reentry_vec = np.zeros((N4_reentry+1,1))
        x4_reentry_sol[:,0] = rocket_return.unscale_x(x4_before_reentry_scaled_sol).full().flatten()
        t4_reentry_vec[0] = t4_ballistic_vec[-1]
        alt4_reentry_sol[0] = ca.norm_2(ca.vertcat(x4_reentry_sol[0,0], rocket_return.R0+x4_reentry_sol[1,0])) - rocket_return.R0
        vel4_reentry_sol[0] = ca.norm_2(ca.vertcat(x4_reentry_sol[2,0], x4_reentry_sol[3,0]))
        Qdyn4_reentry_sol[0] = 0.5 * atmosphere.rho_fun(alt4_reentry_sol[0]) * vel4_reentry_sol[0]**2
        for k in range(N4_reentry):
            t4_reentry_vec[k+1] = t4_reentry_vec[k] + dt4_reentry_sol/N4_reentry
            x4_reentry_sol[:,k+1] = rocket_return.unscale_x(rocket_return.dynamics_kp1(rocket_return.scale_x(x4_reentry_sol[:,k]), u4_reentry_scaled_sol, dt4_reentry_sol/N4_reentry)).full().flatten()
            alt4_reentry_sol[k+1] = ca.norm_2(ca.vertcat(x4_reentry_sol[0,k+1], rocket_return.R0+x4_reentry_sol[1,k+1])) - rocket_return.R0
            vel4_reentry_sol[k+1] = ca.norm_2(ca.vertcat(x4_reentry_sol[2,k+1], x4_reentry_sol[3,k+1]))
            Qdyn4_reentry_sol[k+1] = 0.5 * atmosphere.rho_fun(alt4_reentry_sol[k+1]) * vel4_reentry_sol[k+1]**2
            hflux4_reentry_sol[k+1] = k_empirical*np.sqrt(atmosphere.rho_fun(alt4_reentry_sol[k+1])) * vel4_reentry_sol[k+1]**3

        # before landing phase
        N4_before_landing = 50
        x4_before_landing_vec_sol = np.zeros((rocket_return.nx, N4_before_landing+1))
        alt4_before_landing_sol, vel4_before_landing_sol = np.zeros((N4_before_landing+1,1)), np.zeros((N4_before_landing+1,1))
        Qdyn4_before_landing_sol, hflux4_before_landing_sol = np.zeros((N4_before_landing+1,1)), np.zeros((N4_before_landing+1,1))
        t4_before_landing_vec = np.zeros((N4_before_landing+1,1))
        x4_before_landing_vec_sol[:,0] = rocket_return.unscale_x(x4_after_reentry_scaled_sol).full().flatten()
        t4_before_landing_vec[0] = t4_reentry_vec[-1]
        alt4_before_landing_sol[0] = rocket_return.local_to_alt(x4_before_landing_vec_sol[:,0])
        vel4_before_landing_sol[0] = ca.norm_2(ca.vertcat(x4_before_landing_vec_sol[2,0], x4_before_landing_vec_sol[3,0]))
        Qdyn4_before_landing_sol[0] = 0.5 * atmosphere.rho_fun(alt4_before_landing_sol[0]) * vel4_before_landing_sol[0]**2
        hflux4_before_landing_sol[0] = k_empirical * np.sqrt(atmosphere.rho_fun(alt4_before_landing_sol[0])) * vel4_before_landing_sol[0]**3
        for k in range(N4_before_landing):
            t4_before_landing_vec[k+1] = t4_before_landing_vec[k] + dt4_before_landing_sol/N4_before_landing
            x4_before_landing_vec_sol[:,k+1] = rocket_return.unscale_x(rocket_return.dynamics_kp1(rocket_return.scale_x(x4_before_landing_vec_sol[:,k]), 0, dt4_before_landing_sol/N4_before_landing)).full().flatten()
            alt4_before_landing_sol[k+1] = rocket_return.local_to_alt(x4_before_landing_vec_sol[:,k+1])
            vel4_before_landing_sol[k+1] = ca.norm_2(ca.vertcat(x4_before_landing_vec_sol[2,k+1], x4_before_landing_vec_sol[3,k+1]))
            Qdyn4_before_landing_sol[k+1] = 0.5 * atmosphere.rho_fun(alt4_before_landing_sol[k+1]) * vel4_before_landing_sol[k+1]**2
            hflux4_before_landing_sol[k+1] = k_empirical * np.sqrt(atmosphere.rho_fun(alt4_before_landing_sol[k+1])) * vel4_before_landing_sol[k+1]**3
        
        # landing burn phase
        N4_landing = N4
        alt4_landing_sol, vel4_landing_sol = np.zeros((N4_landing+1,1)), np.zeros((N4_landing+1,1))
        Qdyn4_landing_sol = np.zeros((N4_landing+1,1))
        t4_landing_vec = np.linspace(t4_before_landing_vec[-1], t4_before_landing_vec[-1] + dt4_landing_sol*N4_landing, N4_landing+1)
        for k in range(N4_landing+1):
            alt4_landing_sol[k] = rocket_return.local_to_alt(x4_landing_sol[:,k])
            vel4_landing_sol[k] = ca.norm_2(ca.vertcat(x4_landing_sol[2,k], x4_landing_sol[3,k]))
            Qdyn4_landing_sol[k] = 0.5 * atmosphere.rho_fun(alt4_landing_sol[k]) * vel4_landing_sol[k]**2

    print(f'Payload Mass: {payload_mass_sol:.2f} kg')
    print(f'Launch Azimuth: {np.rad2deg(LaunchAz_sol):.2f} deg')
    print(f'dV for final orbit: {v3_desired-v3_apogee:.2f} m/s')
    print(f'Propellent mass for final dV: {propellent_mass_for_final_dv3:.2f} kg')
    print(f'Final Trajectory Inclination (Target): {np.rad2deg(i3_sol[-1]):.2f} deg (Target: {np.rad2deg(Target_Orbit["i"]):.2f} deg)')
    print(f'Final Trajectory Apogee (Target): {(actual_apogee3 - rocket_eci.R0)/1000:.2f} m (Target: {(Target_Orbit["apogee"] - rocket_eci.R0)/1000:.2f} m)')
    print(f'Final Trajectory Perigee (Target): {(apogee3_sol[-1] - rocket_eci.R0)/1000:.2f} m (Target: {(Target_Orbit["perigee"] - rocket_eci.R0)/1000:.2f} m)')
    print(f'Parking Orbit (Apogee, Perigee, Inclination): {(apogee3_sol[-1] - rocket_eci.R0)/1000:.2f} Km, {(perigee3_sol[-1] - rocket_eci.R0)/1000:.2f} Km, {np.rad2deg(i3_sol[-1]):.2f} deg')


    
    # plot the results
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f'dt1={dt1_sol:.2f} s, dt2={dt2_sol:.2f} s, dt3={dt3_sol:.2f} s, LaunchAz={np.rad2deg(LaunchAz_sol):.2f} deg, PayloadMass={payload_mass_sol:.2f} kg', fontsize=16)
    gs = gridspec.GridSpec(3, 4, figure=fig)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.plot(x1_sol[0,:]/1000, x1_sol[1,:]/1000, label='1st Stage')
    if RecoveryStrategy != 'None':
        if RecoveryStrategy == 'ASDS':
            x_earth = np.linspace(0, 850000,1000)
            z_earth = np.sqrt(rocket_return.R0**2 - x_earth**2) - rocket_return.R0
            ax_traj.plot(x_earth/1000, z_earth/1000, 'k--')

        ax_traj.plot(x4_ballistic_sol[0,:]/1000, x4_ballistic_sol[1,:]/1000, label='Booster Return')
        ax_traj.plot(x4_reentry_sol[0]/1000, x4_reentry_sol[1]/1000)
        ax_traj.plot(x4_reentry_sol[0,:]/1000, x4_reentry_sol[1,:]/1000, label='Reentry Burn')
        ax_traj.plot(x4_before_landing_vec_sol[0,:]/1000, x4_before_landing_vec_sol[1,:]/1000)
        ax_traj.plot(x4_landing_sol[0,:]/1000, x4_landing_sol[1,:]/1000, label='Landing Burn')
        ax_traj.plot(x4_before_reentry_sol[0]/1000, x4_before_reentry_sol[1]/1000, 's')
        if RecoveryStrategy == 'RTLS':
            ax_traj.plot(x4_boostback_vec_sol[0,:]/1000, x4_boostback_vec_sol[1,:]/1000, label='Boostback Phase')
            ax_traj.plot(x4_boostback_sol[0]/1000, x4_boostback_sol[1]/1000, 's', label='Booster Return')
    ax_traj.set_title('Trajectory')
    ax_traj.set_ylabel('Altitude [Km]')
    ax_traj.set_xlabel('Distance [Km]')
    ax_traj.grid()
    ax_traj.set_aspect('equal', adjustable='box')

    ax_alt = fig.add_subplot(gs[0,1])
    ax_alt.plot(t1_vec, alt1_sol/1000, label='1st Stage')
    ax_alt.plot(t2_vec, alt2_sol/1000, label='2nd Stage before fairing separation')
    ax_alt.plot(t3_vec, alt3_sol/1000, label='2nd Stage after fairing separation')
    if RecoveryStrategy != 'None':
        ax_alt.plot(t4_ballistic_vec, alt4_ballistic_sol/1000, label='Booster Return')
        ax_alt.plot(t4_reentry_vec, alt4_reentry_sol/1000, label='Reentry Burn')
        ax_alt.plot(t4_before_landing_vec, alt4_before_landing_sol/1000, label='Landing Burn')
        ax_alt.plot(t4_landing_vec, alt4_landing_sol/1000, label='Landing Burn')
        ax_alt.plot(t4_reentry_vec[0], alt4_before_reentry_sol[0]/1000, 's')
    if RecoveryStrategy == 'RTLS':
        ax_alt.plot(t4_boostback_vec, alt4_boostback_sol/1000, label='Boostback')

    ax_alt.set_title('Altitude')
    ax_alt.set_ylabel('Altitude [Km]')
    ax_alt.set_xlabel('Time [s]')
    ax_alt.grid()

    ax_incl = fig.add_subplot(gs[1,0])
    ax_incl.plot(t1_vec, np.rad2deg(i1_sol), label='1st Stage')
    ax_incl.plot(t2_vec, np.rad2deg(i2_sol), label='2nd Stage before fairing separation')
    ax_incl.plot(t3_vec, np.rad2deg(i3_sol), label='2nd Stage after fairing separation')
    ax_incl.plot([0, t3_vec[-1]], [np.rad2deg(Target_Orbit['i']+0.001), np.rad2deg(Target_Orbit['i']+0.001)], 'k--')
    ax_incl.plot([0, t3_vec[-1]], [np.rad2deg(Target_Orbit['i']-0.001), np.rad2deg(Target_Orbit['i']-0.001)], 'k--')
    ax_incl.plot([0, t3_vec[-1]], [np.rad2deg(Target_Orbit['i']), np.rad2deg(Target_Orbit['i'])], 'k--')
    ax_incl.set_title('Inclination')
    ax_incl.set_ylabel('Inclination [deg]')
    ax_incl.set_xlabel('Time [s]')
    ax_incl.grid()

    ax_ap = fig.add_subplot(gs[1,1])
    ax_ap.plot(t1_vec, (perigee1_sol - rocket_eci.R0)/1000, label='Perigee')
    ax_ap.plot(t2_vec, (perigee2_sol - rocket_eci.R0)/1000)
    ax_ap.plot(t3_vec, (perigee3_sol - rocket_eci.R0)/1000)
    ax_ap.plot(t1_vec, (apogee1_sol - rocket_eci.R0)/1000, label='Apogee')
    ax_ap.plot(t2_vec, (apogee2_sol - rocket_eci.R0)/1000)
    ax_ap.plot(t3_vec, (apogee3_sol - rocket_eci.R0)/1000)
    ax_ap.plot([0, t3_vec[-1]], [(Target_Orbit['apogee'] - rocket_eci.R0)/1000, (Target_Orbit['apogee'] - rocket_eci.R0)/1000], 'k--')
    ax_ap.plot([0, t3_vec[-1]], [(Target_Orbit['perigee'] - rocket_eci.R0)/1000, (Target_Orbit['perigee'] - rocket_eci.R0)/1000], 'k--')
    ax_ap.set_title('Apogee/Perigee Altitude')
    ax_ap.set_ylabel('Apogee/Perigee [Km]')
    ax_ap.set_xlabel('Time [s]')
    ax_ap.grid()
    ax_ap.legend()
    ax_ap.set_yscale('symlog', linthresh=100)


    ax_vel = fig.add_subplot(gs[0,2])
    ax_vel.plot(t1_vec, np.linalg.norm(x1_sol[2:4,:], axis=0), label='1st Stage Local')
    ax_vel.plot(t2_vec, np.linalg.norm(x2_sol[3:6,:], axis=0), label='2nd Stage ECI')
    ax_vel.plot(t3_vec, np.linalg.norm(x3_sol[3:6,:], axis=0), label='2nd Stage ECI')
    if RecoveryStrategy == 'RTLS':
        ax_vel.plot(t4_boostback_vec, vel4_boostback_sol, label='Boostback')
    if RecoveryStrategy != 'None':
        ax_vel.plot(t4_ballistic_vec, vel4_ballistic_sol, label = 'Booster Local')
        ax_vel.plot(t4_reentry_vec, vel4_reentry_sol)
        ax_vel.plot(t4_before_landing_vec, vel4_before_landing_sol)
        ax_vel.plot(t4_landing_vec, vel4_landing_sol)
        ax_vel.plot(t4_reentry_vec[0], vel4_before_reentry_sol[0], 's')
    ax_vel.set_title('Velocity')
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.set_xlabel('Time [s]')
    ax_vel.grid()
    ax_throttle = fig.add_subplot(gs[1,2])
    ax_throttle.plot(t1_vec[:-1], u1_sol[0,:])
    ax_throttle.plot(t2_vec[:-1], np.linalg.norm(u2_sol, axis=0)/rocket_eci.SecondStage_Thrust)
    ax_throttle.plot(t3_vec[:-1], np.linalg.norm(u3_sol, axis=0)/rocket_eci.SecondStage_Thrust)
    if RecoveryStrategy != 'None':
        ax_throttle.plot(t4_ballistic_vec, 0*t4_ballistic_vec, label='Ballistic Phase')
        ax_throttle.plot(t4_reentry_vec, np.ones_like(t4_reentry_vec)*u4_reentry_scaled_sol, label='Reentry Phase')
        ax_throttle.plot(t4_landing_vec[:-1], u4_landing_scaled_sol.T, label='Landing Phase')
    ax_throttle.set_title('Thrust Factor')
    ax_throttle.set_ylabel('Thrust Factor [-]')
    ax_throttle.set_xlabel('Time [s]')
    ax_throttle.grid()
    ax_throttle.set_ylim(0, 1.1)
    ax_alpha = fig.add_subplot(gs[2,2])
    ax_alpha.plot(t1_vec[:-1], np.rad2deg(u1_sol[1,:]))
    ax_alpha.plot(t2_vec[:-1], np.rad2deg(Alpha2_sol))
    ax_alpha.plot(t3_vec[:-1], np.rad2deg(Alpha3_sol))
    ax_alpha.set_title('Angle of Attack')
    ax_alpha.set_ylabel('Angle of Attack [deg]')
    ax_alpha.set_xlabel('Time [s]')
    ax_alpha.grid()
    ax_mass = fig.add_subplot(gs[2,0])
    ax_mass.plot(t1_vec, x1_sol[4,:]/1000)
    ax_mass.plot(t2_vec, x2_sol[6,:]/1000)
    ax_mass.plot(t3_vec, x3_sol[6,:]/1000)
    if RecoveryStrategy == 'RTLS':
        ax_mass.plot(t4_boostback_vec, x4_boostback_vec_sol[4,:]/1000, label='Boostback')
        ax_mass.plot(dt4_boostback_sol, x4_boostback_sol[4]/1000,'s', label='Booster Return')
    if RecoveryStrategy != 'None':
        ax_mass.plot(t4_ballistic_vec, x4_ballistic_sol[4,:]/1000)
        ax_mass.plot(t4_reentry_vec, x4_reentry_sol[4,:]/1000)
        ax_mass.plot(t4_before_landing_vec, x4_before_landing_vec_sol[4,:]/1000)
        ax_mass.plot(t4_landing_vec, x4_landing_sol[4,:]/1000)
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.FirstStage_EmptyMass)/1000, (payload_mass_sol+booster.FirstStage_EmptyMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.LV_total_mass)/1000, (payload_mass_sol+booster.LV_total_mass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.SecondStage_FullMass)/1000, (payload_mass_sol+booster.SecondStage_FullMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.SecondStage_EmptyMass)/1000, (payload_mass_sol+booster.SecondStage_EmptyMass)/1000], 'k--')
    ax_mass.plot(t3_vec[-1], (x3_sol[6,rocket_eci.N[1]]-propellent_mass_for_final_dv_sol)/1000, 's', label='Mass After Final dV')
    if RecoveryStrategy != 'None':
        ax_mass.plot([0, t4_landing_vec[-1,0]], [(booster.EmptyFirstStageMass)/1000, (booster.EmptyFirstStageMass)/1000], 'k--')
    if optimize_stage_partition:
        ax_mass.set_title(f'Optimal First Stage fraction - {FirstStage_Propellent_fraction_sol:2.4}')
    ax_mass.set_ylabel('Mass [Tons]')
    ax_mass.set_xlabel('Time [s]')
    ax_mass.grid()
    ax_acc = fig.add_subplot(gs[1,3])
    ax_acc.plot(t1_vec[:-1], acc1_sol)
    ax_acc.plot(t2_vec[:-1], acc2_sol)
    ax_acc.plot(t3_vec[:-1], acc3_sol)
    ax_acc.set_title('Specific Acceleration')
    ax_acc.set_ylabel('Specific Acceleration [m/s^2]')
    ax_acc.set_xlabel('Time [s]')
    ax_acc.grid()
    ax_q = fig.add_subplot(gs[2,1])
    ax_q.plot(t1_vec, Qdyn1_sol/1000.0, label='1st Stage')
    if RecoveryStrategy != 'None':
        ax_q.plot(t4_ballistic_vec, Qdyn4_ballistic_sol/1000.0, label='Booster Return')
        ax_q.plot(t4_reentry_vec, Qdyn4_reentry_sol/1000.0, label='Reentry Burn')
        ax_q.plot(t4_before_landing_vec, Qdyn4_before_landing_sol/1000.0, label='Landing Burn')
        ax_q.plot(t4_landing_vec, Qdyn4_landing_sol/1000.0, label='Landing Burn')

        x4_before_landing_intren_sol = x4_after_reentry_scaled_sol
        for i in range(N_before_landing_constraints):
            x4_before_landing_intren_sol = rocket_return.dynamics_kp1(x4_before_landing_intren_sol, 0, dt4_before_landing_sol/N_before_landing_constraints)
            x4_before_landing_intren_sol_unscaled = rocket_return.unscale_x(x4_before_landing_intren_sol)
            Qdyn = 0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren_sol_unscaled)) * (x4_before_landing_intren_sol_unscaled[2]**2 + x4_before_landing_intren_sol_unscaled[3]**2)
            ax_q.plot(t4_reentry_vec[-1] + (i+1)*dt4_before_landing_sol/N_before_landing_constraints, Qdyn/1000.0, 'ro')

    ax_q.set_title('Qynamic Pressure')
    ax_q.set_ylabel('Qynamic Pressure [KPa]')
    ax_q.grid()

    ax_gama = fig.add_subplot(gs[2,3])
    ax_gama.plot(t1_vec, np.rad2deg(gama1_sol), label='1st Stage')
    ax_gama.plot(t2_vec, np.rad2deg(gama2_sol), label='2nd Stage before fairing separation')
    ax_gama.plot(t3_vec, np.rad2deg(gama3_sol), label='2nd Stage after fairing separation') 
    ax_gama.set_title('Gama')
    ax_gama.set_ylabel('Gama [deg]')
    ax_gama.set_xlabel('Time [s]')
    ax_gama.grid()
    ax_gama.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))

    ax_heat = fig.add_subplot(gs[0,3])
    # ax_heat.plot(t1_vec, 2*Qdyn1_sol*vel1_sol/1e6.0, label='1st Stage')
    if RecoveryStrategy != 'None':
        ax_heat.plot(t4_reentry_vec, hflux4_reentry_sol/1000.0, label='Reentry Burn')
        ax_heat.plot(t4_before_landing_vec, hflux4_before_landing_sol/1000.0, label='Landing Burn')

        x4_before_landing_intren_sol = x4_after_reentry_scaled_sol
        for i in range(N_before_landing_constraints):
            x4_before_landing_intren_sol = rocket_return.dynamics_kp1(x4_before_landing_intren_sol, 0, dt4_before_landing_sol/N_before_landing_constraints)
            x4_before_landing_intren_sol_unscaled = rocket_return.unscale_x(x4_before_landing_intren_sol)
            hflux = k_empirical * np.sqrt(atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren_sol_unscaled))) * np.linalg.norm(x4_before_landing_intren_sol_unscaled[2:4])**3
            ax_heat.plot(t4_reentry_vec[-1] + (i+1)*dt4_before_landing_sol/N_before_landing_constraints, hflux/1000.0, 'ro')

    ax_heat.set_title('Empirical Heat Flux')
    ax_heat.set_ylabel('Empirical Heat Flux [KWa/m^2]')
    ax_heat.grid()


    plt.tight_layout()

    # # Calc parking orbit
    # x_parking = x3_sol[:,-1].reshape(1,-1)
    # dtheta = 0
    # tf = 0
    # while True:
    #     x_parking = np.vstack((x_parking, rocket_eci.dynamics_kp1(x_parking[-1,:], 0, 1.0).full().flatten()))
    #     tf += 1.0
    #     dtheta += np.arccos(np.dot(x_parking[-1,0:3],x_parking[-2,0:3])/(np.linalg.norm(x_parking[-1,0:3])*np.linalg.norm(x_parking[-2,0:3])))
    #     if dtheta >= 2*np.pi:
    #         break

    # r_parking = np.linalg.norm(x_parking[:,0:3], axis=1)
    # i_apogee = np.argmax(r_parking)
    # x_final = x_parking[i_apogee,:].reshape(1,-1)
    # x_final[-1,3:6] = v3_desired * x_final[-1,3:6] / np.linalg.norm(x_final[-1,3:6])

    # Parking_apogee = np.linalg.norm(x_parking[np.argmax(r_parking),0:3])
    # Parking_perigee = np.linalg.norm(x_parking[np.argmin(r_parking),0:3])

    
    # dtheta = 0
    # while True:
    #     x_final = np.vstack((x_final, rocket_eci.dynamics_kp1(x_final[-1,:], 0, dt3_sol).full().flatten()))
    #     dtheta += np.arccos(np.dot(x_final[-1,0:3],x_final[-2,0:3])/(np.linalg.norm(x_final[-1,0:3])*np.linalg.norm(x_final[-2,0:3])))
    #     if dtheta >= 2*np.pi:
    #         break
    # r_final = np.linalg.norm(x_final[:,0:3], axis=1)

    # final_apogee = np.linalg.norm(x_final[np.argmax(r_final),0:3])
    # final_perigee = np.linalg.norm(x_final[np.argmin(r_final),0:3])
        


    # phi, theta = np.mgrid[0:np.pi:30j, 0:2*np.pi:30j]
    # x = rocket_Booster.R0 * np.sin(phi) * np.cos(theta)
    # y = rocket_Booster.R0 * np.sin(phi) * np.sin(theta)
    # z = rocket_Booster.R0 * np.cos(phi)

    # # Create equatorial plane (z=0)
    # theta_plane = np.linspace(0, 2*np.pi, 100)
    # r_plane = np.linspace(0, rocket_Booster.R0*1.1, 25)
    # r_grid, theta_grid = np.meshgrid(r_plane, theta_plane)
    # x_plane = r_grid * np.cos(theta_grid)
    # y_plane = r_grid * np.sin(theta_grid)
    # z_plane = np.zeros_like(x_plane)
    
    # fig_3d = plt.figure(figsize=(12, 12))
    # ax_3d = plt.subplot(1,2,1, projection='3d')

    # # Plot Earth
    # ax_3d.plot_surface(x, y, z, rstride=2, cstride=2, color='blue', alpha=0.1, edgecolor='none', antialiased=False)

    # # Plot equatorial plane
    # ax_3d.plot_surface(x_plane, y_plane, z_plane, color='gray', alpha=0.4)

    # #plot trajectory
    # ax_3d.plot(x1_eci_sol[0,:], x1_eci_sol[1,:], x1_eci_sol[2,:], 'r')
    # ax_3d.plot(x2_sol[0,:], x2_sol[1,:], x2_sol[2,:], 'g')
    # ax_3d.plot(x3_sol[0,:], x3_sol[1,:], x3_sol[2,:], 'g')
    # ax_3d.plot(x_parking[:,0], x_parking[:,1], x_parking[:,2], 'c', label=f'Parking Orbit (apogee={np.round((np.max(r_parking) - rocket_Booster.R0)/1000)}, perigee={np.round((np.min(r_parking) - rocket_Booster.R0)/1000)})')
    # ax_3d.plot(x_final[:,0], x_final[:,1], x_final[:,2], '--m', label=f'Final Orbit (apogee={np.round((np.max(final_apogee) - rocket_Booster.R0)/1000)}, perigee={np.round((np.min(final_perigee) - rocket_Booster.R0)/1000)})')
    # ax_3d.legend()

    # # ECI axes
    # max_range = rocket_Booster.R0 * 2
    # ax_3d.quiver(0, 0, 0, max_range, 0, 0, color='r')
    # ax_3d.quiver(0, 0, 0, 0, max_range, 0, color='g')
    # ax_3d.quiver(0, 0, 0, 0, 0, max_range, color='b')

    # # Set aspect and limits
    # ax_3d.set_aspect('equal')
    # ax_3d.set_xlabel('X (km)')
    # ax_3d.set_ylabel('Y (km)')
    # ax_3d.set_zlabel('Z (km)')
    # ax_3d.set_title('Earth in ECI Frame with Equatorial Plane')
    # ax_3d.legend()

    # ax_traj3d = plt.subplot(1,2,2, projection='3d')
    #     #plot trajectory
    # ax_traj3d.plot(x1_eci_sol[0,:], x1_eci_sol[1,:], x1_eci_sol[2,:], 'r', label='1st Stage')
    # ax_traj3d.plot(x2_sol[0,:], x2_sol[1,:], x2_sol[2,:], 'c', label='2nd Stage before fairing separation')
    # ax_traj3d.plot(x3_sol[0,:], x3_sol[1,:], x3_sol[2,:], 'g', label='2nd Stage after fairing separation')
    # for i in range(0,u2_sol.shape[1], 5):
    #     u_vec = u2_sol[:,i] / np.linalg.norm(u2_sol[:,i])*100000
    #     ax_traj3d.quiver(x2_sol[0,i], x2_sol[1,i], x2_sol[2,i], u_vec[0], u_vec[1], u_vec[2], color='b')
    # for i in range(0,u3_sol.shape[1], 5):
    #     u_vec = u3_sol[:,i] / np.linalg.norm(u3_sol[:,i])*100000
    #     ax_traj3d.quiver(x3_sol[0,i], x3_sol[1,i], x3_sol[2,i], u_vec[0], u_vec[1], u_vec[2], color='b')

    
    # ax_traj3d.legend()


    # with open('GPS III SV01 raw.json', 'r') as f:
    #     data = json.load(f)

    # with open('stage2 raw.json', 'r') as f:
    #     data2 = json.load(f)
    # with open('analysed.json', 'r') as f:
    #     analyzed = json.load(f)

    # ax_alt.plot(data['time'], data['altitude'],'k--', label='GPS III SV01')
    # ax_alt.plot(data2['time'], data2['altitude'],'c--', label='Stage 2')

    # ax_vel.plot(data['time'], data['velocity'],'k--')
    # ax_vel.plot(data2['time'], data2['velocity'],'c--')

    # ax_q.plot(analyzed['time'], np.array(analyzed['q'])/1000,'k--')

    # ax_acc.plot(analyzed['time'], analyzed['acceleration'],'k--')

    plt.show()

