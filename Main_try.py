#!/usr/bin/env python3

import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.gridspec as gridspec
import casadi as ca
import pickle

import Atmosphere_Type as Atm_Type
import LV_Type as LV_Type




if __name__ == '__main__':
    # Create an instance of the AtmosphereType class
    atmosphere = Atm_Type.AtmosphereType()
    # Create an instance of the VLType class
    rocket_Booster = LV_Type.BoosterLaunchVehicle_2D(atmosphere=atmosphere)
    rocket_eci = LV_Type.LaunchVehicle_ECI()
    rocket_return = LV_Type.BoosterReturn_2D(atmosphere=atmosphere)
    # Target orbit parameters for LEO
    Target_Orbit = {"apogee": rocket_Booster.R0 + 400.0*1000.0, # semi-major axis
                    "perigee": rocket_Booster.R0 + 200.0*1000.0, # semi-minor axis
                    "i": np.deg2rad(60.0), # inclination [rad]
                    }

    # Target orbit parameters for GTO
    # Target_Orbit = {"apogee": rocket_Booster.R0 + 35786.0*1000.0, # semi-major axis
    #                 "perigee": rocket_Booster.R0 + 200.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(0.0), # inclination [rad]
    #                 }

    # Target orbit parameters for SSO
    # Target_Orbit = {"apogee": rocket_Booster.R0 + 780 * 1000.0, # semi-major axis
    #                 "perigee": rocket_Booster.R0 + 780.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(98.6), # inclination [rad]
    #                 }

    Target_Orbit["a"] = 0.5 * (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # semi-major axis
    Target_Orbit["e"] = (Target_Orbit["apogee"] - Target_Orbit["perigee"]) / (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # eccentricity

    RecoveryStrategy = 'None' # 'None', 'RTLS', 'ASDS'
    # define the optimization problem
    opti = ca.Opti()

    # First Phase - Booster 
    N1 = 60
    payload_mass = opti.variable(1)
    x1 = opti.variable(rocket_Booster.nx, N1+1) # [x, z, vx, vz, m]
    u1 = opti.variable(rocket_Booster.nu, N1) # [T_factor, alpha]
    dt1 = opti.variable(1)
    LaunchAz = opti.variable(1)

    # Set the initial state
    init_time = 5.0
    init_acc = rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.LV_total_mass + payload_mass) - rocket_Booster.g0 * (rocket_Booster.R0 / (rocket_Booster.R0 + rocket_Booster.LaunchAltitude))**2
    init_vz = init_acc * init_time
    init_vx = 0.0
    init_z = 0.5 * init_acc * init_time**2
    init_x = 0.0
    init_m = rocket_Booster.LV_total_mass + payload_mass - (rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.FirstStage_SL_Isp * rocket_Booster.g0)) * init_time

    opti.subject_to(x1[:,0] == ca.vertcat(init_x, init_z, init_vx, init_vz, init_m))
    opti.subject_to(payload_mass >= 0.0)
    opti.subject_to(dt1 >= 0.1)
    # set the constraints:
    for k in range(N1):
        # set the dynamics
        # opti.subject_to(x1[:,k+1] == rocket_Booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))
        opti.subject_to((x1[:,k+1]-rocket_Booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))/dt1 == 0.0)
        # set the thrust constraints
        opti.subject_to(u1[0,k] <= 1.0)
        opti.subject_to(u1[0,k]/rocket_Booster.FirstStage_MinThrust_Factor >= 1.0)
        opti.subject_to(u1[1,k]/rocket_Booster.FirstStage_MaxAlpha <= 1.0)
        opti.subject_to(-u1[1,k]/rocket_Booster.FirstStage_MaxAlpha <= 1.0)

        # set the state constraints
        opti.subject_to(0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)<=rocket_Booster.FirstStage_MaxDynamicPressure)
        # opti.subject_to(0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)/rocket_Booster.FirstStage_MaxDynamicPressure <= 1.0)
        opti.subject_to(u1[1,k]*0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)<=1000.0)
        # opti.subject_to(-u1[1,k]*0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)<=1000.0)
        # opti.subject_to(u1[1,k]*0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)/1000.0 <= 1.0)
        # opti.subject_to(-u1[1,k]*0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)/1000.0 <= 1.0)
    # set the final state constraints
    # opti.subject_to(x1[4,N1] >= rocket_Booster.FirstStage_EmptyMass + payload_mass + dm)
    opti.subject_to(x1[4,N1]/(rocket_Booster.FirstStage_EmptyMass + payload_mass) >= 1.0)

    # Phase 2 - Second Stage, before fairing separation (if needed)
    N2 = 10
    x2 = opti.variable(rocket_eci.nx, N2+1) # [x, y, z, vx, vy, vz, m]
    u2 = opti.variable(rocket_eci.nu, N2) # [fx, fy, fz]
    dt2 = opti.variable(1)

    # Set the initial state - the final state of the boost phase transformed to ECI
    # opti.subject_to(LaunchAz >= -0.9*np.pi)
    # opti.subject_to(LaunchAz <= 0.9*np.pi)
    opti.subject_to(dt2*N2 >= 10.0)
    eci_pos, eci_vel = rocket_eci.local_to_eci_func(ca.vertcat(x1[0,N1], 0.0, x1[1,N1]), ca.vertcat(x1[2,N1], 0.0, x1[3,N1]), LaunchAz)
    opti.subject_to(x2[0:3,0] == eci_pos)
    opti.subject_to(x2[3:6,0] == eci_vel)
    opti.subject_to(x2[6,0] == rocket_eci.SecondStage_FullMass + payload_mass)
    for k in range(N2):
        # set the dynamics
        # opti.subject_to(x2[:,k+1] == rocket_eci.dynamics_kp1(x2[:,k], u2[:,k], dt2))
        opti.subject_to((x2[:,k+1]-rocket_eci.dynamics_kp1(x2[:,k], u2[:,k], dt2))/dt2 == 0.0)
        # set the thrust constraints
        # opti.subject_to(ca.norm_2(u2[:,k]) <= rocket_eci.SecondStage_Thrust)
        opti.subject_to(ca.norm_2(u2[:,k])/rocket_eci.SecondStage_Thrust <= 1.0)
        # set the time step constraints

    # set the final state constraints
    # opti.subject_to(rocket_eci.eci_to_alt(x2[:,N2]) >= rocket_eci.FairingSeparationAltitude)
    opti.subject_to(rocket_eci.eci_to_alt(x2[:,N2])/rocket_eci.FairingSeparationAltitude >= 1.0)

    ## phase 3 - Second Stage, after fairing separation
    N3 = 60
    x3 = opti.variable(rocket_eci.nx, N3+1) # [x, y, z, vx, vy, vz, m]
    u3 = opti.variable(rocket_eci.nu, N3)
    dt3 = opti.variable(1)
    # Set the initial state - the final state of the boost phase transformed to ECI
    opti.subject_to(dt3 >= 0.5)
    opti.subject_to(x3[0:6,0] == x2[0:6,N2])
    opti.subject_to(x3[6,0]/(x2[6,N2] - rocket_eci.FairingMass) == 1.0)
    for k in range(N3):
        # set the dynamics
        # opti.subject_to(x3[:,k+1] == rocket_eci.dynamics_kp1(x3[:,k], u3[:,k], dt3*(1-0.5*k/(N3-1))))
        # opti.subject_to(x3[:,k+1] == rocket_eci.dynamics_kp1(x3[:,k], u3[:,k], dt3))
        opti.subject_to((x3[:,k+1]-rocket_eci.dynamics_kp1(x3[:,k], u3[:,k], dt3))/dt3 == 0.0)
        # set the thrust constraints
        # opti.subject_to(ca.norm_2(u3[:,k]) == rocket_eci.SecondStage_Thrust)
        opti.subject_to(ca.norm_2(u3[:,k])/rocket_eci.SecondStage_Thrust == 1.0)
        # set the time step constraints

    # set the final state constraints (Parking orbit)
    h = ca.cross(x3[0:3,N3], x3[3:6,N3])
    cos_i = h[2] / ca.norm_2(h)
    i = ca.acos(ca.fmin(ca.fmax(cos_i, -1.0), 1.0))
    # i = ca.acos(h[2] / ca.norm_2(h))
    e = ca.cross(x3[3:6,N3], h) / (rocket_eci.mu) - x3[0:3,N3] / ca.norm_2(x3[0:3,N3])
    eps = 0.5*ca.sumsqr(x3[3:6,N3]) - rocket_eci.mu / ca.norm_2(x3[0:3,N3])
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
    propellent_mass_for_final_dv = x3[6,N3] * (1.0 - ca.exp(-ca.fabs(v_apogee - v_desired) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))
    
    # v_final_apogee = ca.sqrt(rocket_eci.mu * (1.0 / Target_Orbit["a"]))
    # dv_for_inclination_fix = 2 * v_apogee * ca.sin(Target_Orbit["i"]/2.0)
    # propellent_mass_for_inclination_fix = (x3[6,N3]-propellent_mass_for_final_dv) * (1.0 - ca.exp(-ca.fabs(dv_for_inclination_fix) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))


    # propellent_mass_for_final_dv = 0.0

    opti.subject_to(x3[6,N3]/(rocket_eci.SecondStage_EmptyMass + payload_mass + propellent_mass_for_final_dv) >= 1.0)
    opti.subject_to(apogee/Target_Orbit["perigee"] == 1.0)
    opti.subject_to(perigee/(rocket_eci.R0 + 100*1000.0) >= 1.0)

    # Phase 4 - Booster Return:
    if RecoveryStrategy != 'None':
        N4 = 15
        x4_0 = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_before_reentry = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        x4_after_reentry = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        u4_reentry = opti.variable(rocket_return.nu, 1) # [T_factor]
        # x4_before_landing = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
        # x4_landing = opti.variable(rocket_return.nx, N4+1) # [x, z, vx, vz, m]
        # u4_landing = opti.variable(rocket_return.nu, N4) # [T_factor]
        dt4_reentry = opti.variable(1)
        dt4_ballistic = opti.variable(1)
        # dt4_before_landing = opti.variable(1)
        # dt4_landing = opti.variable(1)


        # Set the initial state
        opti.subject_to(x4_0[0:4,0] == x1[0:4,N1])
        opti.subject_to(x4_0[4,0]/(x1[4,N1] - rocket_eci.SecondStage_FullMass - payload_mass) == 1.0)

        # set the time step constraints
        opti.subject_to(dt4_ballistic/100.0 >= 1.0)
        opti.subject_to(dt4_reentry/10 >= 1.0)
        # opti.subject_to(dt4_before_landing/20.0 >= 1.0)
        # opti.subject_to(dt4_landing*N4/10.0 >= 1.0)

        # Ballistic phase
        opti.subject_to((x4_before_reentry-rocket_return.dynamics_kp1_M50(x4_0, 0, dt4_ballistic))/200.0 == 0.0)
        # opti.subject_to(rocket_return.local_to_alt(x4_before_reentry)/60000.0 >= 1.0)
        # opti.subject_to(rocket_return.local_to_alt(x4_before_reentry)/65000.0 <= 1.0)

        # Reentry phase - Dynamics of entry burn
        opti.subject_to((x4_after_reentry-rocket_return.dynamics_kp1_M20(x4_before_reentry, u4_reentry, dt4_reentry))/200.0 == 0.0)
        # set the thrust constraints
        opti.subject_to(u4_reentry <= 1.0)
        opti.subject_to(u4_reentry/rocket_return.FirstStage_MinThrust_Factor >=  1.0)
        opti.subject_to(ca.norm_2(ca.vertcat(x4_after_reentry[2], x4_after_reentry[3]))/1400.0 <= 1.0)
        opti.subject_to(rocket_return.local_to_alt(x4_after_reentry)/45000.0 == 1.0)

        # After Reentry, before landing burn
        # opti.subject_to((x4_before_landing-rocket_return.dynamics_kp1_M20(x4_after_reentry, 0, dt4_before_landing))/200.0 == 0.0)
        # opti.subject_to(rocket_return.local_to_alt(x4_before_landing)/1000.0 == 1.0)
        # opti.subject_to(rocket_return.local_to_alt(x4_before_landing) <= 2000.0)

        # N_before_landing_constraints = 2
        # x4_before_landing_intren = x4_reentry
        # for _ in range(N_before_landing_constraints):
        #     x4_before_landing_intren = rocket_return.dynamics_kp1_M20(x4_before_landing_intren, 0, dt4_before_landing/N_before_landing_constraints)
        #     opti.subject_to(0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren)) * ca.norm_2(ca.vertcat(x4_before_landing_intren[2], x4_before_landing_intren[3]))**2 <= rocket_return.Booster_MaxDynamicPressure)
            # opti.subject_to(0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren)) * ca.norm_2(ca.vertcat(x4_before_landing_intren[2], x4_before_landing_intren[3]))**3 <= rocket_return.Booster_MaxHeatFlux)

            # Landing Burn:
        # opti.subject_to(x4_landing[:,0] == x4_before_landing)
        # for k in range(N4):
        #     opti.subject_to((x4_landing[:,k+1]-rocket_return.dynamics_kp1(x4_landing[:,k], u4_landing[:,k], dt4_landing))/200.0 == 0.0)
                
        # opti.subject_to(u4_landing[k] * 9.0 <= 1.0)
        # opti.subject_to(u4_landing[k] * 9.0/rocket_return.FirstStage_MinThrust_Factor >= 1.0)
            
        # if RecoveryStrategy == 'RTLS':
        #     opti.subject_to(x4_landing[0,N4] == 0.0)
        #     opti.subject_to(x4_landing[1,N4] == 0.0)
        #     opti.subject_to(x4_landing[2,N4] == 0.0)
        #     opti.subject_to(x4_landing[3,N4] == 0.0)
        # elif RecoveryStrategy == 'ASDS':
        #     opti.subject_to(ca.norm_2(x4_landing[2:4,N4]) == 1.0) # zero velocity at the end of the flight
        #     # opti.subject_to(rocket_return.local_to_alt(x4_landing[:,N4]) <= 10.0) # landing altitude
        #     opti.subject_to(rocket_return.local_to_alt(x4_landing[:,N4]) == 0.0) # landing altitude
        # opti.subject_to(x4_landing[4,N4]/rocket_Booster.EmptyFirstStageMass == 1.0) # mass at landing

    # set the cost function
    cost = 0.0
    # cost = 0.5*(ca.sumsqr(u1[0,:]) + ca.sumsqr(u1[1,:])/rocket_Booster.FirstStage_MaxAlpha**2)
    cost += 0.5*(ca.sumsqr(u1[0,1:]-u1[0,:-1]) + ca.sumsqr(u1[1,1:]-u1[1,:-1])/rocket_Booster.FirstStage_MaxAlpha**2 )
    # cost += 0.5*(ca.sumsqr(u2) + ca.sumsqr(u3))/(rocket_eci.SecondStage_Thrust)**2
    cost += -0.5*payload_mass - 0.001*x3[6,N3]
    cost += ca.sumsqr((i-Target_Orbit["i"])/np.deg2rad(0.5))
    if RecoveryStrategy != 'None':
        # cost += 0.5*(ca.sumsqr(u4_landing[1:]-u4_landing[:-1]) + ca.sumsqr(u4_reentry) )/N4
        # cost += (dt4_before_landing)*0.001
        # cost += -x4_landing[4,N4]/rocket_Booster.EmptyFirstStageMass # mass at landing
        # cost += ca.sumsqr(u4_landing[1:] - u4_landing[:-1]) # minimize the thrust changes
        cost += ca.sumsqr(u4_reentry)
        # cost += (ca.norm_2(x4_landing[2:4,N4])-1)**2 # mass at landing
        # cost += (x4_landing[4,N4] - rocket_Booster.EmptyFirstStageMass)**2
    opti.minimize(cost)


    # set the solver
    opts = {
            "print_time": 0,  # Print timing, 
            "ipopt": {
            "tol": 1e-4,  # Convergence tolerance
            # "max_iter": 1000,  # Max iterations
            # "print_level": 0,  # Verbosity level
            # "alpha_for_y": "min",  # Fraction-to-boundary rule parameter
            # "timing_statistics": "yes", # Enable timing statistics
            # "nlp_scaling_method": "None", # Scaling method
            # 'nlp_scaling_max_gradient': 100,
            # "obj_scaling_factor": 1e-2, # Scaling factor for the objective function
            "mu_strategy": "adaptive",  # "monotone" or "adaptive" Strategy for updating the barrier parameter
            # "mu_init": 1e-1,  # 1e-1 Initial value of the barrier parameter
            # "mu_min": 1e-11,  # 1e-11 Smallest mu allowed — optimization stops when mu ≤ this
            # "mu_target": 0.0,  # 0.0 IPOPT drives mu towards this target; if 0, tries to reach exact KKT solution
            # "barrier_tol_factor": 2.0,  # Affects how mu and tolerances interact during convergence
            # "alpha_for_y": "min",  # inf, 1.0 Maximum step size for the line search
        }}
    opti.solver('ipopt', opts)

    # load the initial guess
    with open('solution.pickle', 'rb') as f:
        solution = pickle.load(f)
    with open('ReturnSolution.pickle', 'rb') as f:
        return_solution = pickle.load(f)

    # calculate the initial guess for first stage 
    x1_guess = np.zeros((rocket_Booster.nx, N1+1))
    u1_guess = np.zeros((rocket_Booster.nu, N1))
    dt1_guess = 148.0 / N1
    payload_mass_guess = 5000.0
    LaunchAz_guess = np.pi/2.0 - Target_Orbit["i"] + np.deg2rad(rocket_Booster.LaunchLatitude)

    init_acc_guess = rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.LV_total_mass + payload_mass_guess) - rocket_Booster.g0 * (rocket_Booster.R0 / (rocket_Booster.R0 + rocket_Booster.LaunchAltitude))**2
    init_vz_guess = init_acc_guess * init_time
    init_vx_guess = 0.0
    init_z_guess = 0.5 * init_acc_guess * init_time**2
    init_x_guess = 0.0
    init_m_guess = rocket_Booster.LV_total_mass + payload_mass_guess - (rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.FirstStage_SL_Isp * rocket_Booster.g0)) * init_time
    x0_guess = np.array([init_x_guess, init_z_guess, init_vx_guess, init_vz_guess, init_m_guess])

    x1_guess[:,0] = x0_guess
    iter=0
    while iter < 10:
        for k in range(N1):
            if x1_guess[1,k]<5000 or x1_guess[1,k]>10000:
                u1_guess[0,k] = 1.0
            else:
                u1_guess[0,k] = 0.70
            if np.atan2(x1_guess[3,k], x1_guess[2,k]) > 88.0*np.pi/180.0 and np.linalg.norm(x1_guess[2:4,k]) > 75.0:
                u1_guess[1,k] = -np.deg2rad(1.0)
            else:
                u1_guess[1,k] = 0.0
            
            alt = rocket_Booster.local_to_alt(x1_guess[:,k])
            Q = 0.5 * (x1_guess[2,k]**2 + x1_guess[3,k]**2) * atmosphere.rho_fun(alt)
            x1_guess[:,k+1] = rocket_Booster.dynamics_kp1(x1_guess[:,k], u1_guess[:,k], dt1_guess).full().flatten()
        if rocket_Booster.local_to_alt(x1_guess[:,N1]) < 60000:
            dt1_guess += 0.01
            iter += 1
        else:
            break

    Q_guess = np.zeros((N1+1))
    alt1_guess = np.zeros((N1+1))
    for k in range(N1+1):
        alt1_guess[k] = ca.norm_2(ca.vertcat(x1_guess[0,k], rocket_Booster.R0 + x1_guess[1,k])) - rocket_Booster.R0
        Q_guess[k] = 0.5 * (x1_guess[2,k]**2 + x1_guess[3,k]**2) * atmosphere.rho_fun(alt1_guess[k])
    

    # calculate the initial guess for second stage before fairing separation, if needed
    dt2_guess = 50.0 / N2
    x2_guess = np.zeros((rocket_eci.nx, N2+1))
    u2_guess = np.zeros((rocket_eci.nu, N2))
    pos, vel = rocket_eci.local_to_eci_func(ca.vertcat(x1_guess[0,N1], 0.0, x1_guess[1,N1]), ca.vertcat(x1_guess[2,N1], 0.0, x1_guess[3,N1]), LaunchAz_guess)
    x2_guess[0:3,0] = pos.full().flatten()
    x2_guess[3:6,0] = vel.full().flatten()
    x2_guess[6,0] = rocket_eci.SecondStage_FullMass + payload_mass_guess
    iter=0
    while iter < 10:
        for k in range(N2):
            u2_guess[0:3,k] = (x2_guess[3:6,k] / ca.norm_2(x2_guess[3:6,k])).full().flatten() * rocket_eci.SecondStage_Thrust
            x2_guess[:,k+1] = rocket_eci.dynamics_kp1(x2_guess[:,k], u2_guess[:,k], dt2_guess).full().flatten()
            alt = ca.norm_2(x2_guess[0:3,k+1]) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
        if abs(alt-rocket_eci.FairingSeparationAltitude) < 100.0:
            break
        else:
            alt_N1 = ca.norm_2(ca.vertcat(x1_guess[0,N1], rocket_Booster.R0 + x1_guess[1,N1])) - rocket_Booster.R0
            dt2_guess = dt2_guess*(rocket_eci.FairingSeparationAltitude-alt_N1)/ (alt-alt_N1)
            iter += 1

# calculate the initial guess for second stage after fairing separation
    dt3_guess = 370.0/N3
    x3_guess = np.zeros((rocket_eci.nx, N3+1))
    u3_guess = np.zeros((rocket_eci.nu, N3))
    x3_guess[:,0] = x2_guess[:,N2]
    iter=0
    
    alpha3_init = 0*np.deg2rad(10.0)
    v3_max = ca.sqrt(rocket_eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["apogee"]))
    while iter < 50:
        alpha3, alpha_factor = alpha3_init, 1.0
        for k in range(N3):
            alt = np.linalg.norm(x3_guess[0:3,k]) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
            h = np.cross(x3_guess[0:3,k], x3_guess[3:6,k])
            h = h / np.linalg.norm(h)
            i = np.arccos(h[2] / np.linalg.norm(h))
            u3_guess[0:3,k] = (x3_guess[3:6,k] / np.linalg.norm(x3_guess[3:6,k])) 
            alpha_factor = k/N3
            alpha3 = -alpha3_init*alpha_factor
            u3_guess[0:3,k] = u3_guess[0:3,k] * np.cos(alpha3) + np.cross(h, u3_guess[0:3,k]) * np.sin(alpha3) + h * np.dot(h, u3_guess[0:3,k]) * (1 - np.cos(alpha3)) 
            
            u3_guess[0:3,k] *= rocket_eci.SecondStage_Thrust
            # x3_guess[:,k+1] = rocket_eci.dynamics_kp1(x3_guess[:,k], u3_guess[:,k], dt3_guess*(1-0.5*k/(N3-1))).full().flatten()
            x3_guess[:,k+1] = rocket_eci.dynamics_kp1(x3_guess[:,k], u3_guess[:,k], dt3_guess).full().flatten()
            h = np.cross(x3_guess[0:3,k+1], x3_guess[3:6,k+1])
            e = ca.cross(x3_guess[3:6,k+1], h) / (rocket_eci.mu) - x3_guess[0:3,k+1] / ca.norm_2(x3_guess[0:3,k+1])
            eps = 0.5*ca.sumsqr(x3_guess[3:6,k+1]) - rocket_eci.mu / ca.norm_2(x3_guess[0:3,k+1])
            a = -rocket_eci.mu / (2.0*eps)
            perigee = a * (1.0 - ca.norm_2(e))
            apogee = a * (1.0 + ca.norm_2(e))
            i = ca.acos(h[2] / ca.norm_2(h))
            
            if x3_guess[6,k+1] < rocket_eci.SecondStage_EmptyMass + payload_mass_guess:
                break
        if k < N3-1:
            iter += 1
            dt3_guess = dt3_guess * (k+1) / N3
        elif x3_guess[6,k+1] < rocket_eci.SecondStage_EmptyMass + payload_mass_guess:
            dt3_guess -= 0.25
            iter += 1
        # elif perigee < rocket_eci.R0+100*10**3:
        #     dt3_guess += 0.1
        #     iter += 1
        elif apogee > rocket_eci.R0+1000*10**3 or np.linalg.norm(x3_guess[3:6,N3]) > v3_max+100.0:
            dt3_guess -= 0.025
            iter += 1
        elif perigee < rocket_eci.R0+10*10**3:
            alpha3_init += np.deg2rad(1.0)
            # dt3_guess += 0.
            iter += 1
        else:
            break

    opti.set_initial(dt3, dt3_guess)
    for k in range(N3+1):
        opti.set_initial(x3[:,k], x3_guess[:,k])
    for k in range(N3):
        opti.set_initial(u3[:,k], u3_guess[:,k])

    opti.set_initial(payload_mass, payload_mass_guess)
    opti.set_initial(LaunchAz, LaunchAz_guess)
    opti.set_initial(dt1, dt1_guess)
    opti.set_initial(dt2, dt2_guess)
    opti.set_initial(dt3, dt3_guess)
    opti.set_initial(x1, x1_guess)
    opti.set_initial(u1, u1_guess)
    opti.set_initial(x2, x2_guess)
    opti.set_initial(u2, u2_guess)
    opti.set_initial(x3, x3_guess)
    opti.set_initial(u3, u3_guess)

    # calculate the initial guess for booster return
    if RecoveryStrategy != 'None':
        # calculate the initial guess for booster return
        dt4_ballistic_guess = 200
        x4_0_guess = np.zeros((rocket_return.nx, 1))
        x4_0_guess[0:4,0] = x1_guess[0:4,N1]
        x4_0_guess[4,0] = x1_guess[4,N1] - rocket_eci.SecondStage_FullMass - payload_mass_guess
        
        iter=0
        while iter < 50:
            x4_before_reentry_guess = rocket_return.dynamics_kp1_M50(x4_0_guess, 0, dt4_ballistic_guess)
            alt = ca.norm_2(ca.vertcat(x4_before_reentry_guess[0,0], rocket_return.R0+x4_before_reentry_guess[1,0])) - rocket_return.R0
            if abs(alt - 60000) > 10:
                dt4_ballistic_guess += np.clip((alt - 60000)/abs(x4_before_reentry_guess[3,0]), -10.0, 10.0)
                iter += 1
            else:
                break
        
        u4_reentry_guess = 0.3333
        dt4_reentry_guess = 12.0
        iter = 0
        while iter < 50:
            x4_after_reentry_guess = rocket_return.dynamics_kp1_M20(x4_before_reentry_guess, u4_reentry_guess, dt4_reentry_guess)
            alt = ca.norm_2(ca.vertcat(x4_after_reentry_guess[0,0], rocket_return.R0+x4_after_reentry_guess[1,0])) - rocket_return.R0
            vel = ca.norm_2(ca.vertcat(x4_after_reentry_guess[2,0], x4_after_reentry_guess[3,0]))
            mass = x4_after_reentry_guess[4,0]
            if abs(vel- 1250.0) > 10:
                dt4_reentry_guess += 0.05 * np.sign(vel - 1250.0)
                iter += 1
            else:
                break

        # after reentry burn before landing burn
        iter = 0
        dt4_before_landing_guess = 50.0
        alt4_before_landing = 1250.0
        while iter < 50:
            x4_before_landing_guess = rocket_return.dynamics_kp1_M20(x4_after_reentry_guess, 0, dt4_before_landing_guess)
            alt = ca.norm_2(ca.vertcat(x4_before_landing_guess[0,0], rocket_return.R0+x4_before_landing_guess[1,0])) - rocket_return.R0
            vel = ca.norm_2(ca.vertcat(x4_before_landing_guess[2,0], x4_before_landing_guess[3,0]))
            if abs(alt - alt4_before_landing) > 10:
                dt4_before_landing_guess += 0.9*(alt - alt4_before_landing)/vel
                iter += 1
            else:
                break

        # initial guess for the landing burn
        dt4_landing_guess = 30 / N4
        x4_landing_guess = np.zeros((rocket_return.nx, N4+1))
        u4_landing_guess = np.zeros((rocket_return.nu, N4))
        iter = 0
        thrust_factor = 1.0
        while iter < 1:
            u4_landing_guess[0,0] = 0.1111
            x4_landing_guess[:,0] = x4_before_landing_guess.full().flatten()
            for k in range(N4):
                vel = ca.norm_2(ca.vertcat(x4_landing_guess[2,k], x4_landing_guess[3,k]))
                if vel > 75.0:
                    u4_landing_guess[0,k] = 0.1111 * thrust_factor
                else:
                    u4_landing_guess[0,k] = x4_landing_guess[4,k] * rocket_return.g0/rocket_return.FirstStage_SL_Thrust
                x4_landing_guess[:,k+1] = rocket_return.dynamics_kp1(x4_landing_guess[:,k], u4_landing_guess[0,k], dt4_landing_guess).full().flatten()
            alt = rocket_return.local_to_alt(x4_landing_guess[:,k+1])
            vel = ca.norm_2(ca.vertcat(x4_landing_guess[2,k+1], x4_landing_guess[3,k+1]))
            if abs(alt) > 20:
                dt4_landing_guess += 0.2*(alt)/vel/N4
                iter += 1
            elif abs(vel) > 20:
                thrust_factor += 0.5*vel/10
                iter += 1
            else:
                break
        t4_landing_guess = np.array(dt4_ballistic_guess + dt4_reentry_guess + dt4_before_landing_guess) + np.linspace(0, dt4_landing_guess*N4, N4+1)
        t4_landing_guess = t4_landing_guess.reshape((-1,))

        
        opti.set_initial(x4_0, x4_0_guess)
        opti.set_initial(dt4_ballistic, dt4_ballistic_guess)
        opti.set_initial(dt4_reentry, dt4_reentry_guess)
        # opti.set_initial(dt4_before_landing, dt4_before_landing_guess)
        # opti.set_initial(dt4_landing, dt4_landing_guess)
        opti.set_initial(x4_before_reentry, x4_before_reentry_guess)
        opti.set_initial(x4_after_reentry, x4_after_reentry_guess)
        opti.set_initial(u4_reentry, u4_reentry_guess)
        # opti.set_initial(x4_before_landing, x4_before_landing_guess)
        # opti.set_initial(x4_landing, x4_landing_guess)
        # opti.set_initial(u4_landing, u4_landing_guess)


    # Solution from file:
    if solution['Target_Orbit'] == Target_Orbit:
        opti.set_initial(payload_mass, solution['payload_mass'])
        opti.set_initial(LaunchAz, solution['LaunchAz'])
        opti.set_initial(dt1, solution['dt1'])
        opti.set_initial(dt2, solution['dt2'])
        opti.set_initial(dt3, solution['dt3'])
        opti.set_initial(x1, solution['x1'])
        opti.set_initial(u1, solution['u1'])
        opti.set_initial(x2, solution['x2'])
        opti.set_initial(u2, solution['u2'])
        opti.set_initial(x3, solution['x3'])
        opti.set_initial(u3, solution['u3'])
        if RecoveryStrategy != 'None':
            opti.set_initial(x4_0, return_solution['x4_0'])
            opti.set_initial(dt4_ballistic, return_solution['dt4_ballistic'])
            opti.set_initial(dt4_reentry, return_solution['dt4_reentry'])
            # opti.set_initial(dt4_before_landing, return_solution['dt4_before_landing'])
            # opti.set_initial(dt4_landing, return_solution['dt4_landing'])
            opti.set_initial(x4_before_reentry, return_solution['x4_before_reentry'])
            opti.set_initial(x4_after_reentry, return_solution['x4_after_reentry'])
            opti.set_initial(u4_reentry, return_solution['u4_reentry'])
            # opti.set_initial(x4_before_landing, return_solution['x4_before_landing'])
            # opti.set_initial(x4_landing, return_solution['x4_landing'])
            # opti.set_initial(u4_landing, return_solution['u4_landing'])


    if RecoveryStrategy != 'None':
        dt4_landing_sol = 0.0
        x4_landing_sol = np.zeros((rocket_return.nx, N4+1))
        u4_landing_sol = np.zeros((rocket_return.nu, N4))
    
    # Solve the optimization problem
    try:
        sol = opti.solve()

        payload_mass_sol = np.array(sol.value(payload_mass))
        LaunchAz_sol = np.array(sol.value(LaunchAz))
        u1_sol = np.array(sol.value(u1))
        x1_sol = np.array(sol.value(x1))
        dt1_sol = np.array(sol.value(dt1))
        u2_sol = np.array(sol.value(u2))
        x2_sol = np.array(sol.value(x2))
        dt2_sol = np.array(sol.value(dt2))
        dt3_sol = np.array(sol.value(dt3))
        x3_sol = np.array(sol.value(x3))
        u3_sol = np.array(sol.value(u3))
        propellent_mass_for_final_dv_sol = np.array(sol.value(propellent_mass_for_final_dv))
        if RecoveryStrategy != 'None':
            x4_0_sol = np.array(sol.value(x4_0))
            dt4_ballistic_sol = np.array(sol.value(dt4_ballistic))
            x4_before_reentry_sol = np.array(sol.value(x4_before_reentry))
            x4_after_reentry_sol = np.array(sol.value(x4_after_reentry))
            u4_reentry_sol = np.array(sol.value(u4_reentry))
            dt4_reentry_sol = np.array(sol.value(dt4_reentry))
            # dt4_before_landing_sol = np.array(sol.value(dt4_before_landing))
            # x4_before_landing_sol = np.array(sol.value(x4_before_landing))
            # dt4_landing_sol = np.array(sol.value(dt4_landing))
            # x4_landing_sol = np.array(sol.value(x4_landing))
            # u4_landing_sol = np.array(sol.value(u4_landing))

    except:

        g_expr = opti.debug.g  # Symbolic expression for all constraints
        g_func = ca.Function("g_func", [opti.x], [g_expr])
        x_val = opti.debug.value(opti.x)
        g_val = g_func(x_val).full().flatten()

        lbg = opti.debug.lbg
        ubg = opti.debug.ubg # Evaluates g at solution

        tol = 1e-3
        print("Significant constraint residuals (> 1e-3):")
        for i, val in enumerate(g_val):
            lb = lbg[i]
            ub = ubg[i]
            if val < lb - tol or val > ub + tol:
                # print(f"  g[{i}] = {val:.4e}, bounds = [{lb:.4e}, {ub:.4e}]")
                print(f"  g[{i}] = {val:.4e}")

        f_expr = opti.f  # symbolic expression for cost
        f_func = ca.Function("f_func", [opti.x], [f_expr])
        f_val = f_func(opti.debug.value(opti.x)).full().flatten()
        print(f"True objective value at solution: {f_val[0]:.6e}")

        g = opti.debug.g
        J = ca.jacobian(g, opti.x)
        J_func = ca.Function("J", [opti.x], [J])
        J_val = J_func(opti.debug.value(opti.x))
        print("Jacobian max:", np.max(np.abs(J_val.full())))



        payload_mass_sol = opti.debug.value(payload_mass)
        LaunchAz_sol = opti.debug.value(LaunchAz)
        u1_sol = opti.debug.value(u1)
        x1_sol = opti.debug.value(x1)
        dt1_sol = opti.debug.value(dt1)
        u2_sol = opti.debug.value(u2)
        x2_sol = opti.debug.value(x2)
        dt2_sol = opti.debug.value(dt2)
        dt3_sol = opti.debug.value(dt3)
        x3_sol = opti.debug.value(x3)
        u3_sol = opti.debug.value(u3)
        propellent_mass_for_final_dv_sol = np.array(opti.debug.value(propellent_mass_for_final_dv))
        if RecoveryStrategy != 'None':
            x4_0_sol = opti.debug.value(x4_0)
            dt4_ballistic_sol = opti.debug.value(dt4_ballistic)
            x4_before_reentry_sol = opti.debug.value(x4_before_reentry)
            x4_after_reentry_sol = opti.debug.value(x4_after_reentry)
            u4_reentry_sol = opti.debug.value(u4_reentry)
            dt4_reentry_sol = opti.debug.value(dt4_reentry)
            # dt4_before_landing_sol = opti.debug.value(dt4_before_landing)
            # x4_before_landing_sol = opti.debug.value(x4_before_landing)
            # dt4_landing_sol = opti.debug.value(dt4_landing)
            # x4_landing_sol = opti.debug.value(x4_landing)
            # u4_landing_sol = opti.debug.value(u4_landing)

        grad_f = ca.gradient(opti.f, opti.x)
        grad_func = ca.Function("grad_f", [opti.x], [grad_f])
        g_val = grad_func(opti.debug.value(opti.x))
        for i, val in enumerate(g_val.full().flatten()):
            if val > 1e-1:
                print(f"  g[{i}] = {val:.4e}")
    
    t1_vec = np.linspace(init_time, init_time+N1*dt1_sol, N1+1)
    t2_vec = np.linspace(t1_vec[-1], t1_vec[-1]+N2*dt2_sol, N2+1)
    t3_vec = np.linspace(t2_vec[-1], t2_vec[-1]+N3*dt3_sol, N3+1)
    # t3_vec = np.zeros((N3+1))
    # t3_vec[0] = t2_vec[-1]
    # for k in range(N3):
    #     t3_vec[k+1] = t3_vec[k] + dt3_sol*(1-0.5*k/(N3-1))

    Isp1_sol = np.zeros((N1))
    Qdyn1_sol = np.zeros((N1+1))
    a1_sol, b1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    apogee1_sol, perigee1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    alt1_sol, vel1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    i1_sol, gama1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    for k in range(N1+1):
        if k < N1:
            Isp1_sol[k] = rocket_Booster.ISP_calc(x1_sol[:,k], u1_sol[:,k]).full().flatten()[0]
        Qdyn1_sol[k] = 0.5 * (x1_sol[2,k]**2 + x1_sol[3,k]**2) * atmosphere.rho_fun(x1_sol[1,k])
        pos, vel = rocket_eci.local_to_eci_func(ca.vertcat(x1_sol[0,k], 0.0, x1_sol[1,k]), ca.vertcat(x1_sol[2,k], 0.0, x1_sol[3,k]), LaunchAz_sol)
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

    Isp2_sol = np.zeros((N2))
    Alpha2_sol = np.zeros((N2))
    a2_sol, b2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    apogee2_sol, perigee2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    alt2_sol, i2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    Qdyn2_sol, gama2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    for k in range(N2+1):
        if k < N2:
            Alpha2_sol[k] = np.arccos(np.dot(u2_sol[:,k], x2_sol[3:6,k]) / (rocket_eci.SecondStage_Thrust * ca.norm_2(x2_sol[3:6,k])))
            Isp2_sol[k] = rocket_eci.ISP_calc(x2_sol[:,k], u2_sol[:,k]).full().flatten()[0]
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

    Isp3_sol = np.zeros((N3))
    Alpha3_sol = np.zeros((N3))
    a3_sol, b3_sol = np.zeros((N3+1)), np.zeros((N3+1))
    apogee3_sol, perigee3_sol = np.zeros((N3+1)), np.zeros((N3+1))
    alt3_sol, i3_sol, gama3_sol = np.zeros((N3+1)), np.zeros((N3+1)), np.zeros((N3+1))
    for k in range(N3+1):
        if k < N3:
            Alpha3_sol[k] = np.arccos(np.dot(u3_sol[:,k], x3_sol[3:6,k]) / (rocket_eci.SecondStage_Thrust * ca.norm_2(x3_sol[3:6,k])))
            Isp3_sol[k] = rocket_eci.ISP_calc(x3_sol[:,k], u3_sol[:,k]).full().flatten()[0]
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
    propellent_mass_for_final_dv3 = x3_sol[6,N3] * (1.0 - ca.exp((v3_apogee - v3_desired) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))
    actual_apogee3 = Target_Orbit["a"] * (1.0 + Target_Orbit["e"])

    if RecoveryStrategy != 'None':
        alt4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[0], rocket_return.R0+x4_0_sol[1])) - rocket_return.R0
        vel4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[2], x4_0_sol[3]))
        N4_ballistic = 50
        x4_ballistic_sol = np.zeros((rocket_return.nx, N4_ballistic))
        alt4_ballistic_sol, vel4_ballistic_sol = np.zeros((N4_ballistic,1)), np.zeros((N4_ballistic,1))
        Qdyn4_ballistic_sol = np.zeros((N4_ballistic,1))
        t4_ballistic_vec = np.zeros((N4_ballistic,1))
        x4_ballistic_sol[:,0] = x4_0_sol
        t4_ballistic_vec[0] = np.linspace(t1_vec[-1], t1_vec[-1]+dt4_ballistic_sol, N4_ballistic+1)[0]
        alt4_ballistic_sol[0] = ca.norm_2(ca.vertcat(x4_ballistic_sol[0,0], rocket_return.R0+x4_ballistic_sol[1,0])) - rocket_return.R0
        vel4_ballistic_sol[0] = ca.norm_2(ca.vertcat(x4_ballistic_sol[2,0], x4_ballistic_sol[3,0]))
        Qdyn4_ballistic_sol[0] = 0.5 * atmosphere.rho_fun(alt4_ballistic_sol[0]) * vel4_ballistic_sol[0]**2
        for k in range(1,N4_ballistic):
            t4_ballistic_vec[k] = t4_ballistic_vec[k-1] + dt4_ballistic_sol/(N4_ballistic-1)
            x4_ballistic_sol[:,k] = rocket_return.dynamics_kp1(x4_ballistic_sol[:,k-1], 0, dt4_ballistic_sol/(N4_ballistic-1)).full().flatten()
            alt4_ballistic_sol[k] = ca.norm_2(ca.vertcat(x4_ballistic_sol[0,k], rocket_return.R0+x4_ballistic_sol[1,k])) - rocket_return.R0
            vel4_ballistic_sol[k] = ca.norm_2(ca.vertcat(x4_ballistic_sol[2,k], x4_ballistic_sol[3,k]))
            Qdyn4_ballistic_sol[k] = 0.5 * atmosphere.rho_fun(alt4_ballistic_sol[k]) * vel4_ballistic_sol[k]**2

        x4_ballistic_M50_sol = rocket_return.dynamics_kp1_M50(x4_0_sol, 0, dt4_ballistic_sol).full().flatten()
        # before reentry burn phase
        alt4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[0], rocket_return.R0+x4_before_reentry_sol[1])) - rocket_return.R0
        vel4_before_reentry_sol = ca.norm_2(ca.vertcat(x4_before_reentry_sol[2], x4_before_reentry_sol[3]))
        Qdyn4_before_reentry_sol = 0.5 * atmosphere.rho_fun(alt4_before_reentry_sol) * vel4_before_reentry_sol**2

        # reentry phase
        N4_reentry = 50
        x4_reentry_sol = np.zeros((rocket_return.nx, N4_reentry))
        alt4_reentry_sol, vel4_reentry_sol = np.zeros((N4_reentry,1)), np.zeros((N4_reentry,1))
        Qdyn4_reentry_sol = np.zeros((N4_reentry,1))
        t4_reentry_vec = np.zeros((N4_reentry,1))
        x4_reentry_sol[:,0] = x4_before_reentry_sol
        t4_reentry_vec[0] = t4_ballistic_vec[-1]
        alt4_reentry_sol[0] = ca.norm_2(ca.vertcat(x4_reentry_sol[0,0], rocket_return.R0+x4_reentry_sol[1,0])) - rocket_return.R0
        vel4_reentry_sol[0] = ca.norm_2(ca.vertcat(x4_reentry_sol[2,0], x4_reentry_sol[3,0]))
        Qdyn4_reentry_sol[0] = 0.5 * atmosphere.rho_fun(alt4_reentry_sol[0]) * vel4_reentry_sol[0]**2
        for k in range(1,N4_reentry):
            t4_reentry_vec[k] = t4_reentry_vec[k-1] + dt4_reentry_sol/(N4_reentry-1)
            x4_reentry_sol[:,k] = rocket_return.dynamics_kp1(x4_reentry_sol[:,k-1], u4_reentry_sol, dt4_reentry_sol/(N4_reentry-1)).full().flatten()
            alt4_reentry_sol[k] = ca.norm_2(ca.vertcat(x4_reentry_sol[0,k], rocket_return.R0+x4_reentry_sol[1,k])) - rocket_return.R0
            vel4_reentry_sol[k] = ca.norm_2(ca.vertcat(x4_reentry_sol[2,k], x4_reentry_sol[3,k]))
            Qdyn4_reentry_sol[k] = 0.5 * atmosphere.rho_fun(alt4_reentry_sol[k]) * vel4_reentry_sol[k]**2

        # before landing phase
        N4_before_landing = 50
        x4_before_landing_vec_sol = np.zeros((rocket_return.nx, N4_before_landing))
        alt4_before_landing_sol, vel4_before_landing_sol = np.zeros((N4_before_landing,1)), np.zeros((N4_before_landing,1))
        Qdyn4_before_landing_sol = np.zeros((N4_before_landing,1))
        t4_before_landing_vec = np.zeros((N4_before_landing,1))
        x4_before_landing_vec_sol[:,0] = x4_reentry_sol[:,-1]
        t4_before_landing_vec[0] = t4_reentry_vec[-1]
        alt4_before_landing_sol[0] = rocket_return.local_to_alt(x4_before_landing_vec_sol[:,0])
        vel4_before_landing_sol[0] = ca.norm_2(ca.vertcat(x4_before_landing_vec_sol[2,0], x4_before_landing_vec_sol[3,0]))
        Qdyn4_before_landing_sol[0] = 0.5 * atmosphere.rho_fun(alt4_before_landing_sol[0]) * vel4_before_landing_sol[0]**2
        for k in range(1,N4_before_landing):
            # t4_before_landing_vec[k] = t4_before_landing_vec[k-1] + dt4_before_landing_sol/(N4_before_landing-1)
            # x4_before_landing_vec_sol[:,k] = rocket_return.dynamics_kp1(x4_before_landing_vec_sol[:,k-1], 0, dt4_before_landing_sol/(N4_before_landing-1)).full().flatten()
            alt4_before_landing_sol[k] = rocket_return.local_to_alt(x4_before_landing_vec_sol[:,k])
            vel4_before_landing_sol[k] = ca.norm_2(ca.vertcat(x4_before_landing_vec_sol[2,k], x4_before_landing_vec_sol[3,k]))
            Qdyn4_before_landing_sol[k] = 0.5 * atmosphere.rho_fun(alt4_before_landing_sol[k]) * vel4_before_landing_sol[k]**2
        
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
    ax_traj.plot(x1_sol[0,:]/1000, alt1_sol/1000, label='1st Stage')
    ax_traj.plot(x1_guess[0,:]/1000, x1_guess[1,:]/1000, 'k--')
    if RecoveryStrategy != 'None':
        x_earth = np.linspace(0, 850000,1000)
        z_earth = np.sqrt(rocket_return.R0**2 - x_earth**2) - rocket_return.R0
        ax_traj.plot(x_earth/1000, z_earth/1000, 'k--')
        ax_traj.plot(x4_ballistic_sol[0,:]/1000, x4_ballistic_sol[1,:]/1000, label='Booster Return')
        ax_traj.plot(x4_reentry_sol[0]/1000, x4_reentry_sol[1]/1000)
        ax_traj.plot(x4_reentry_sol[0,:]/1000, x4_reentry_sol[1,:]/1000, label='Reentry Burn')
        ax_traj.plot(x4_before_landing_vec_sol[0,:]/1000, x4_before_landing_vec_sol[1,:]/1000)
        ax_traj.plot(x4_landing_sol[0,:]/1000, x4_landing_sol[1,:]/1000, label='Landing Burn')
        ax_traj.plot(x4_before_reentry_sol[0]/1000, x4_before_reentry_sol[1]/1000, 's')
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
    ax_ap.set_yscale('symlog', linthresh=1)


    ax_vel = fig.add_subplot(gs[0,2])
    ax_vel.plot(t1_vec, vel1_sol, label='1st Stage')
    ax_vel.plot(t2_vec, np.linalg.norm(x2_sol[3:6,:], axis=0), label='2nd Stage')
    ax_vel.plot(t3_vec, np.linalg.norm(x3_sol[3:6,:], axis=0), label='2nd Stage')
    if RecoveryStrategy != 'None':
        ax_vel.plot(t4_ballistic_vec, vel4_ballistic_sol)
        ax_vel.plot(t4_reentry_vec, vel4_reentry_sol)
        ax_vel.plot(t4_before_landing_vec, vel4_before_landing_sol)
        ax_vel.plot(t4_landing_vec, vel4_landing_sol)
        ax_vel.plot(t4_reentry_vec[0], vel4_before_reentry_sol[0], 's')
    ax_vel.set_title('Velocity ECI')
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.set_xlabel('Time [s]')
    ax_vel.grid()
    ax_acc = fig.add_subplot(gs[1,2])
    ax_acc.plot(t1_vec[:-1], u1_sol[0,:])
    ax_acc.plot(t2_vec[:-1], np.linalg.norm(u2_sol, axis=0)/rocket_eci.SecondStage_Thrust)
    ax_acc.plot(t3_vec[:-1], np.linalg.norm(u3_sol, axis=0)/rocket_eci.SecondStage_Thrust)
    if RecoveryStrategy != 'None':
        ax_acc.plot(t4_ballistic_vec, 0*t4_ballistic_vec, label='Ballistic Phase')
        ax_acc.plot(t4_reentry_vec, np.ones_like(t4_reentry_vec)*u4_reentry_sol, label='Reentry Phase')
        ax_acc.plot(t4_landing_vec[:-1], u4_landing_sol.T, label='Landing Phase')
    ax_acc.set_title('Thrust Factor')
    ax_acc.set_ylabel('Thrust Factor [-]')
    ax_acc.set_xlabel('Time [s]')
    ax_acc.grid()
    ax_acc.set_ylim(0, 1.1)
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
    if RecoveryStrategy != 'None':
        ax_mass.plot(t4_ballistic_vec, x4_ballistic_sol[4,:]/1000)
        ax_mass.plot(t4_reentry_vec, x4_reentry_sol[4,:]/1000)
        ax_mass.plot(t4_before_landing_vec, x4_before_landing_vec_sol[4,:]/1000)
        ax_mass.plot(t4_landing_vec, x4_landing_sol[4,:]/1000)
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.FirstStage_EmptyMass)/1000, (payload_mass_sol+rocket_Booster.FirstStage_EmptyMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.LV_total_mass)/1000, (payload_mass_sol+rocket_Booster.LV_total_mass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.SecondStage_FullMass)/1000, (payload_mass_sol+rocket_Booster.SecondStage_FullMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.SecondStage_EmptyMass)/1000, (payload_mass_sol+rocket_Booster.SecondStage_EmptyMass)/1000], 'k--')
    ax_mass.plot(t3_vec[-1], (x3_sol[6,N3]-propellent_mass_for_final_dv_sol)/1000, 's', label='Mass After Final dV')
    ax_mass.set_title('Mass')
    ax_mass.set_ylabel('Mass [Tons]')
    ax_mass.set_xlabel('Time [s]')
    ax_mass.grid()
    # ax_isp = fig.add_subplot(gs[2,1])
    # ax_isp.plot(t_vec[:-1], Isp_sol)
    # ax_isp.plot([0, t_vec[-1]], [rocket.FirstStage_SL_Isp, rocket.FirstStage_SL_Isp], 'r--')
    # ax_isp.plot([0, t_vec[-1]], [rocket.FirstStage_Vac_Isp, rocket.FirstStage_Vac_Isp], 'k--')
    # ax_isp.set_title('Isp')
    # ax_isp.set_ylabel('Isp [s]')
    # ax_isp.set_xlabel('Time [s]')
    # ax_isp.grid()
    ax_q = fig.add_subplot(gs[2,1])
    ax_q.plot(t1_vec, Qdyn1_sol/1000.0, label='1st Stage')
    # ax_q.plot(t2_vec, Qdyn2_sol/1000.0, label='2nd Stage before fairing separation')
    if RecoveryStrategy != 'None':
        ax_q.plot(t4_ballistic_vec, Qdyn4_ballistic_sol/1000.0, label='Booster Return')
        ax_q.plot(t4_reentry_vec, Qdyn4_reentry_sol/1000.0, label='Reentry Burn')
        ax_q.plot(t4_before_landing_vec, Qdyn4_before_landing_sol/1000.0, label='Landing Burn')
        ax_q.plot(t4_landing_vec, Qdyn4_landing_sol/1000.0, label='Landing Burn')
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
    ax_gama.legend()

    plt.tight_layout()
    plt.show()
        




    