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
import LV_Type_scaled as LV_Type




if __name__ == '__main__':
    # Create an instance of the AtmosphereType class
    atmosphere = Atm_Type.AtmosphereType()
    # Create an instance of the VLType class
    rocket_Booster = LV_Type.BoosterLaunchVehicle_2D(atmosphere=atmosphere)
    rocket_eci = LV_Type.LaunchVehicle_ECI()
    
    # Target orbit parameters for LEO
    Target_Orbit = {"apogee": rocket_Booster.R0 + 300.0*1000.0, # semi-major axis
                    "perigee": rocket_Booster.R0 + 200.0*1000.0, # semi-minor axis
                    "i": np.deg2rad(53.3), # inclination [rad]
                    }

    # Target orbit parameters for GTO
    # Target_Orbit = {"apogee": rocket_Booster.R0 + 35786.0*1000.0, # semi-major axis
    #                 "perigee": rocket_Booster.R0 + 200.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(rocket_Booster.LaunchLatitude), # inclination [rad]
    #                 }

    # Target orbit parameters for SSO
    # Target_Orbit = {"apogee": rocket_Booster.R0 + 780 * 1000.0, # semi-major axis
    #                 "perigee": rocket_Booster.R0 + 780.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(98.6), # inclination [rad]
    #                 }

    Target_Orbit["a"] = 0.5 * (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # semi-major axis
    Target_Orbit["e"] = (Target_Orbit["apogee"] - Target_Orbit["perigee"]) / (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # eccentricity

    dm = 13000.0
    # define the optimization problem
    opti = ca.Opti()

    # First Phase - Booster 
    N1 = 60
    payload_mass_scaled = opti.variable(1) 
    x1 = opti.variable(rocket_Booster.nx, N1+1) # [x, z, vx, vz, m]
    u1 = opti.variable(rocket_Booster.nu, N1) # [T_factor, alpha]
    dt1 = opti.variable(1)
    LaunchAz = opti.variable(1)

    payload_mass = payload_mass_scaled * rocket_Booster.scaleX[4]

    # Set the initial state
    init_time = 5.0
    init_acc = rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.LV_total_mass + payload_mass) - rocket_Booster.g0 * (rocket_Booster.R0 / (rocket_Booster.R0 + rocket_Booster.LaunchAltitude))**2
    init_vz = init_acc * init_time
    init_vx = 0.0
    init_z = 0.5 * init_acc * init_time**2
    init_x = 0.0
    init_m = rocket_Booster.LV_total_mass + payload_mass - (rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.FirstStage_SL_Isp * rocket_Booster.g0)) * init_time

    opti.subject_to(x1[:,0] == rocket_Booster.scale_x(ca.vertcat(init_x, init_z, init_vx, init_vz, init_m)))
    opti.subject_to(payload_mass_scaled >= 0.0)
    opti.subject_to(dt1*N1 >= 60.0)
    # set the constraints:
    for k in range(N1):
        # set the dynamics
        # opti.subject_to(x1[:,k+1] == rocket_Booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))
        opti.subject_to(x1[:,k+1] == rocket_Booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))
        # set the thrust constraints
        opti.subject_to(u1[0,k] <= 1.0)
        opti.subject_to(u1[0,k] >= rocket_Booster.FirstStage_MinThrust_Factor)
        opti.subject_to(u1[1,k] <= 1.0)
        opti.subject_to(-u1[1,k] <= 1.0)

        # set the state constraints
        xk = rocket_Booster.unscale_x(x1[:,k])
        uk = rocket_Booster.unscale_u(u1[:,k])
        q = 0.5 * (xk[2]**2 + xk[3]**2) * atmosphere.rho_fun(rocket_Booster.local_to_alt(xk))

        opti.subject_to(q/rocket_Booster.FirstStage_MaxDynamicPressure <= 1.0) # dynamic pressure limit
        opti.subject_to((uk[1] * q / 1e4)**2 <= 1.0) # alpha * dynamic pressure limit

        # opti.subject_to(0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(xk))*(xk[2]**2+xk[3]**2)<=rocket_Booster.FirstStage_MaxDynamicPressure)
        # opti.subject_to(u1[1,k]*0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)<=1000.0)
        # opti.subject_to(-u1[1,k]*0.5*atmosphere.rho_fun(rocket_Booster.local_to_alt(x1[:,k]))*(x1[2,k]**2+x1[3,k]**2)<=1000.0)
    # set the final state constraints
    # opti.subject_to(x1[4,N1] >= rocket_Booster.FirstStage_EmptyMass + payload_mass + dm)
    
    x1f = rocket_Booster.unscale_x(x1[:,N1])
    q1f = 0.5 * (x1f[2]**2 + x1f[3]**2) * atmosphere.rho_fun(rocket_Booster.local_to_alt(x1f))
    target_mass_scaled = (rocket_Booster.FirstStage_EmptyMass + payload_mass + dm) / rocket_Booster.scaleX[4]
    opti.subject_to(x1[4,N1] >= target_mass_scaled)
    opti.subject_to(q1f/rocket_Booster.FirstStage_StageSeparationMaxDynamicPressure <= 1.0)

    # Phase 2 - Second Stage, before fairing separation (if needed)
    N2 = 10
    x2 = opti.variable(rocket_eci.nx, N2+1) # [x, y, z, vx, vy, vz, m]
    u2 = opti.variable(rocket_eci.nu, N2) # [fx, fy, fz]
    dt2 = opti.variable(1)

    # Set the initial state - the final state of the boost phase transformed to ECI
    # opti.subject_to(LaunchAz >= -0.9*np.pi)
    # opti.subject_to(LaunchAz <= 0.9*np.pi)
    opti.subject_to(dt2*N2 >= 1.0)
    eci_pos, eci_vel = rocket_eci.local_to_eci_func(ca.vertcat(x1f[0], 0.0, x1f[1]), ca.vertcat(x1f[2], 0.0, x1f[3]), LaunchAz)
    x2_init = ca.vertcat(eci_pos, eci_vel, rocket_eci.SecondStage_FullMass + payload_mass)
    opti.subject_to(x2[:,0] == rocket_eci.scale_x(x2_init))
    for k in range(N2):
        # set the dynamics
        opti.subject_to(x2[:,k+1] == rocket_eci.dynamics_kp1(x2[:,k], u2[:,k], dt2))
        # set the thrust constraints
        opti.subject_to(ca.norm_2(u2[:,k]) == 1.0)

    # set the final state constraints
    opti.subject_to(rocket_eci.eci_to_alt(rocket_eci.unscale_x(x2[:,N2]))/rocket_eci.FairingSeparationAltitude >= 1.0)

    ## phase 3 - Second Stage, after fairing separation
    N3 = 60
    x3 = opti.variable(rocket_eci.nx, N3+1) # [x, y, z, vx, vy, vz, m]
    u3 = opti.variable(rocket_eci.nu, N3)
    dt3 = opti.variable(1)
    # Set the initial state - the final state of the boost phase transformed to ECI
    x3_init = ca.vertcat(x2[0:6,N2], x2[6,N2] - rocket_eci.FairingMass/ rocket_eci.scaleX[6])
    opti.subject_to(dt3*N3 >= 60.0)
    opti.subject_to(x3[:,0] == x3_init)
    for k in range(N3):
        # set the dynamics
        opti.subject_to(x3[:,k+1] == rocket_eci.dynamics_kp1(x3[:,k], u3[:,k], dt3))
        # set the thrust constraints
        opti.subject_to(ca.norm_2(u3[:,k]) == 1.0)
        # xk = rocket_eci.unscale_x(x3[:,k])
        # opti.subject_to(rocket_eci.eci_to_alt(xk) >= 0.0)

    # set the final state constraints (Parking orbit)
    x3f = rocket_eci.unscale_x(x3[:,N3])
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
    # propellent_mass_for_inclination_fix = (x3[6,N3]-propellent_mass_for_final_dv) * (1.0 - ca.exp(-ca.fabs(dv_for_inclination_fix) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))


    # propellent_mass_for_final_dv = 0.0
    x3f_mass_scaled = (rocket_eci.SecondStage_EmptyMass + payload_mass + propellent_mass_for_final_dv) / rocket_eci.scaleX[6]
    opti.subject_to(x3[6,N3] >= x3f_mass_scaled)
    opti.subject_to(apogee/Target_Orbit["perigee"] == 1.0)
    opti.subject_to(perigee/(rocket_eci.R0 + rocket_eci.ParkingOrbit_PerigeeAlt) >= 1.0)
    opti.subject_to(i/Target_Orbit["i"] == 1.0)

            
    # set the cost function
    cost = 0.0
    cost += 0.5*10e-3*(ca.sumsqr(u1[0,1:]-u1[0,:-1]) + ca.sumsqr(u1[1,1:]-u1[1,:-1]) )
    cost += -payload_mass_scaled*10
    opti.minimize(cost)


    # set the solver
    opts = {
            # "print_time": 1,  # Print timing, 
            "ipopt": {
            "tol": 1e-8,  # Convergence tolerance
            "max_iter": 100,  # Max iterations
            # "print_level": 0,  # Verbosity level
            # "alpha_for_y": "min",  # Fraction-to-boundary rule parameter
            # "timing_statistics": "yes", # Enable timing statistics
            # "nlp_scaling_method": "None", # 'gradient-based', # Scaling method
            # 'nlp_scaling_max_gradient': 100,
            # "obj_scaling_factor": 1e-4, # Scaling factor for the objective function
            "mu_strategy": "adaptive",  # "monotone" or "adaptive" Strategy for updating the barrier parameter
            # "mu_init": 1e-11,  # 1e-1 Initial value of the barrier parameter
            # "mu_min": 1e-11,  # 1e-11 Smallest mu allowed — optimization stops when mu ≤ this
            # "mu_target": 0.0,  # 0.0 IPOPT drives mu towards this target; if 0, tries to reach exact KKT solution
            # "barrier_tol_factor": 5.0,  # Affects how mu and tolerances interact during convergence
        }}
    opti.solver('ipopt', opts)

    # calculate the initial guess for first stage 
    x1_guess = np.zeros((rocket_Booster.nx, N1+1))
    u1_guess = np.zeros((rocket_Booster.nu, N1))
    dt1_guess = 148.0 / N1
    payload_mass_guess = 9000.0
    LaunchAz_guess = np.pi/2.0 - Target_Orbit["i"]

    init_acc_guess = rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.LV_total_mass + payload_mass_guess) - rocket_Booster.g0 * (rocket_Booster.R0 / (rocket_Booster.R0 + rocket_Booster.LaunchAltitude))**2
    init_vz_guess = init_acc_guess * init_time
    init_vx_guess = 0.0
    init_z_guess = 0.5 * init_acc_guess * init_time**2
    init_x_guess = 0.0
    init_m_guess = rocket_Booster.LV_total_mass + payload_mass_guess - (rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.FirstStage_SL_Isp * rocket_Booster.g0)) * init_time
    x0_guess = np.array([init_x_guess, init_z_guess, init_vx_guess, init_vz_guess, init_m_guess])

    x1_guess[:,0] = rocket_Booster.scale_x(x0_guess).full().flatten()
    iter=0
    while iter < 10:
        for k in range(N1):
            xk = rocket_Booster.unscale_x(x1_guess[:,k])
            uk = rocket_Booster.unscale_u(u1_guess[:,k])
            alt = rocket_Booster.local_to_alt(xk)
            if xk[1]<5000 or xk[1]>10000:
                u1_guess[0,k] = 1.0
            else:
                u1_guess[0,k] = 0.70
            if np.atan2(xk[3], xk[2]) > 88.0*np.pi/180.0 and np.linalg.norm(xk[2:4]) > 75.0:
                u1_guess[1,k] = -np.deg2rad(1.0) / rocket_Booster.FirstStage_MaxAlpha
            else:
                u1_guess[1,k] = 0.0
            
            x1_guess[:,k+1] = rocket_Booster.dynamics_kp1(x1_guess[:,k], u1_guess[:,k], dt1_guess).full().flatten()

        xN = rocket_Booster.unscale_x(x1_guess[:,N1])
        alt_N1 = rocket_Booster.local_to_alt(xN)
        Q = 0.5 * (xN[2]**2 + xN[3]**2) * atmosphere.rho_fun(alt_N1)
        if Q > rocket_Booster.FirstStage_StageSeparationMaxDynamicPressure:
            dt1_guess += 0.5 / N1
            iter += 1
        elif xN[4] <= rocket_Booster.FirstStage_EmptyMass + payload_mass_guess + dm:
            dt1_guess -= 1.0 / N1
            iter += 1
        else:
            break

    # calculate the initial guess for second stage before fairing separation, if needed
    dt2_guess = 50.0 / N2
    x2_guess = np.zeros((rocket_eci.nx, N2+1))
    u2_guess = np.zeros((rocket_eci.nu, N2))
    x1f = rocket_Booster.unscale_x(x1_guess[:,N1])
    pos, vel = rocket_eci.local_to_eci_func(ca.vertcat(x1f[0], 0.0, x1f[1]), ca.vertcat(x1f[2], 0.0, x1f[3]), LaunchAz_guess)
    x2_init = ca.vertcat(pos, vel, rocket_eci.SecondStage_FullMass + payload_mass_guess)
    x2_guess[:,0] = rocket_eci.scale_x(x2_init).full().flatten()
    iter=0
    while iter < 10:
        for k in range(N2):
            xk = rocket_eci.unscale_x(x2_guess[:,k])
            alt = rocket_eci.eci_to_alt(xk)
            u2_guess[0:3,k] = (x2_guess[3:6,k] / ca.norm_2(x2_guess[3:6,k])).full().flatten()
            x2_guess[:,k+1] = rocket_eci.dynamics_kp1(x2_guess[:,k], u2_guess[:,k], dt2_guess).full().flatten()
        xN = rocket_eci.unscale_x(x2_guess[:,N2])
        alt = rocket_eci.eci_to_alt(xN)
        if abs(alt-rocket_eci.FairingSeparationAltitude) < 100.0:
            break
        else:
            dt2_guess = dt2_guess*(rocket_eci.FairingSeparationAltitude-alt_N1)/ (alt-alt_N1)
            iter += 1

# calculate the initial guess for second stage after fairing separation
    dt3_guess = 370.0/N3
    x3_guess = np.zeros((rocket_eci.nx, N3+1))
    u3_guess = np.zeros((rocket_eci.nu, N3))
    x3_guess[0:6,0] = x2_guess[0:6,N2]
    x3_guess[6,0] = x2_guess[6,N2] - rocket_eci.FairingMass / rocket_eci.scaleX[6]
    iter=0
    
    alpha3_init = 0.0
    v3_max = ca.sqrt(rocket_eci.mu * (2.0 / Target_Orbit["perigee"] - 1.0 / Target_Orbit["apogee"]))
    while iter < 50:
        alpha3, alpha_factor = alpha3_init, 1.0
        for k in range(N3):
            xk = rocket_eci.unscale_x(x3_guess[:,k]).full().flatten()
            h = np.cross(xk[0:3], xk[3:6])
            h = h / np.linalg.norm(h)
            u3_guess[0:3,k] = (xk[3:6] / np.linalg.norm(xk[3:6])) 
            # alpha_factor = k/N3
            alpha3 = -alpha3_init*alpha_factor
            u3_guess[0:3,k] = u3_guess[0:3,k] * np.cos(alpha3) + np.cross(h, u3_guess[0:3,k]) * np.sin(alpha3) + h * np.dot(h, u3_guess[0:3,k]) * (1 - np.cos(alpha3)) 
            
            x3_guess[:,k+1] = rocket_eci.dynamics_kp1(x3_guess[:,k], u3_guess[:,k], dt3_guess).full().flatten()
        
            xkp1 = rocket_eci.unscale_x(x3_guess[:,k+1]).full().flatten()
            vel = np.linalg.norm(xkp1[3:6])
            h = np.cross(xkp1[0:3], xkp1[3:6])
            e = ca.cross(xkp1[3:6], h) / (rocket_eci.mu) - xkp1[0:3] / ca.norm_2(xkp1[0:3])
            eps = 0.5*ca.sumsqr(xkp1[3:6]) - rocket_eci.mu / ca.norm_2(xkp1[0:3])
            a = -rocket_eci.mu / (2.0*eps)
            perigee = a * (1.0 - ca.norm_2(e))
            apogee = a * (1.0 + ca.norm_2(e))
            i = ca.acos(h[2] / ca.norm_2(h))
            
            if xkp1[6] < rocket_eci.SecondStage_EmptyMass + payload_mass_guess:
                break
        if k < N3-1:
            iter += 1
            dt3_guess = dt3_guess * (k+1) / N3
        elif xkp1[6] < rocket_eci.SecondStage_EmptyMass + payload_mass_guess:
            dt3_guess -= 0.25
            iter += 1
        elif apogee < rocket_eci.ParkingOrbit_PerigeeAlt + rocket_eci.R0 or a < 0.5*(Target_Orbit['perigee']+rocket_eci.R0+200.0*1e3):
            dt3_guess += 0.1
            iter += 1
        elif perigee < rocket_eci.R0+10*10**3:
            alpha3_init -= np.deg2rad(0.5)
            # dt3_guess += 0.
            iter += 1
        elif apogee > Target_Orbit['perigee'] + 150.0*1e3:
            dt3_guess -= 0.1
            iter += 1
        else:
            break

    # set the initial guess for the optimization variables
    opti.set_initial(payload_mass_scaled, payload_mass_guess/rocket_Booster.scaleX[4])
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

    # Solve the optimization problem
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
        payload_mass_scaled_sol = np.array(sol.value(payload_mass_scaled))
        LaunchAz_sol = np.array(sol.value(LaunchAz))

    except:
        u1_scaled_sol = opti.debug.value(u1)
        x1_scaled_sol = opti.debug.value(x1)
        dt1_scaled_sol = opti.debug.value(dt1)
        u2_scaled_sol = opti.debug.value(u2)
        x2_scaled_sol = opti.debug.value(x2)
        dt2_scaled_sol = opti.debug.value(dt2)
        dt3_scaled_sol = opti.debug.value(dt3)
        x3_scaled_sol = opti.debug.value(x3)
        u3_scaled_sol = opti.debug.value(u3)
        payload_mass_scaled_sol = opti.debug.value(payload_mass_scaled)
        LaunchAz_sol = opti.debug.value(LaunchAz)
        
    payload_mass_sol = payload_mass_scaled_sol * rocket_Booster.scaleX[4]
    dt1_sol = dt1_scaled_sol
    dt2_sol = dt2_scaled_sol
    dt3_sol = dt3_scaled_sol
    u1_sol = np.zeros_like(u1_scaled_sol)
    for i in range(u1_scaled_sol.shape[1]):
        u1_sol[:,i] = rocket_Booster.unscale_u(u1_scaled_sol[:,i]).full().flatten()
    x1_sol = np.zeros_like(x1_scaled_sol)
    for i in range(x1_scaled_sol.shape[1]):
        x1_sol[:,i] = rocket_Booster.unscale_x(x1_scaled_sol[:,i]).full().flatten()
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

    if 0:
        g_expr = opti.debug.g  # Symbolic expression for all constraints
        g_func = ca.Function("g_func", [opti.x], [g_expr])
        x_val = opti.debug.value(opti.x)
        g_val = g_func(x_val).full().flatten()

        lbg = opti.debug.lbg
        ubg = opti.debug.ubg # Evaluates g at solution

        tol = 1e-2
        print("Significant constraint residuals (> 1e-2):")
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

        grad_f = ca.gradient(opti.f, opti.x)
        grad_func = ca.Function("grad_f", [opti.x], [grad_f])
        g_val = grad_func(opti.debug.value(opti.x))
        for i, val in enumerate(g_val.full().flatten()):
            if abs(val) > 1e-1:
                print(f"  g[{i}] = {val:.4e}")


    Isp1_sol = np.zeros((N1))
    Qdyn1_sol = np.zeros((N1+1))
    a1_sol, b1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    apogee1_sol, perigee1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    alt1_sol, vel1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    i1_sol = np.zeros((N1+1))
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

    Isp2_sol = np.zeros((N2))
    Alpha2_sol = np.zeros((N2))
    a2_sol, b2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    apogee2_sol, perigee2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    alt2_sol, i2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    for k in range(N2+1):
        if k < N2:
            Alpha2_sol[k] = np.arccos(np.dot(u2_sol[:,k], x2_sol[3:6,k]) / (rocket_eci.SecondStage_Thrust * ca.norm_2(x2_sol[3:6,k])))
            Isp2_sol[k] = rocket_eci.ISP_calc(x2_sol[:,k], u2_sol[:,k]).full().flatten()[0]
        pos, vel = x2_sol[0:3,k], x2_sol[3:6,k]
        alt2_sol[k] = np.linalg.norm(pos) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
        h2 = ca.cross(pos, vel)
        e2 = ca.cross(vel, h2) / (rocket_eci.mu) - pos / np.linalg.norm(pos)
        eps2 = 0.5*ca.sumsqr(vel) - rocket_eci.mu / np.linalg.norm(pos)
        a2_sol[k] = -rocket_eci.mu / (2.0*eps2)
        b2_sol[k] = a2_sol[k] * np.sqrt(1.0 - ca.sumsqr(e2))
        apogee2_sol[k] = a2_sol[k] * (1.0 + np.linalg.norm(e2))
        perigee2_sol[k] = a2_sol[k] * (1.0 - np.linalg.norm(e2))
        i2_sol[k] = np.arccos(h2[2] / ca.norm_2(h2))

    Isp3_sol = np.zeros((N3))
    Alpha3_sol = np.zeros((N3))
    a3_sol, b3_sol = np.zeros((N3+1)), np.zeros((N3+1))
    apogee3_sol, perigee3_sol = np.zeros((N3+1)), np.zeros((N3+1))
    alt3_sol, i3_sol = np.zeros((N3+1)), np.zeros((N3+1))
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

    v3_apogee = ca.sqrt(rocket_eci.mu * (2.0 / apogee3_sol[-1] - 1.0 / a3_sol[-1]))
    v3_desired = ca.sqrt(rocket_eci.mu * (2.0 / apogee3_sol[-1] - 1.0 / Target_Orbit["a"]))
    propellent_mass_for_final_dv3 = x3_sol[6,N3] * (1.0 - ca.exp((v3_apogee - v3_desired) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))
    actual_apogee3 = Target_Orbit["a"] * (1.0 + Target_Orbit["e"])

    print(f'Payload Mass: {payload_mass_sol:.2f} kg')
    print(f'Launch Azimuth: {np.rad2deg(LaunchAz_sol):.2f} deg')
    print(f'dV for final orbit: {v3_desired-v3_apogee:.2f} m/s')
    print(f'Propellent mass for final dV: {propellent_mass_for_final_dv3:.2f} kg')
    print(f'Final Trajectory Inclination (Target): {np.rad2deg(i3_sol[-1]):.2f} deg (Target: {np.rad2deg(Target_Orbit["i"]):.2f} deg)')
    print(f'Final Trajectory Apogee (Target): {(actual_apogee3 - rocket_eci.R0)/1000:.2f} m (Target: {(Target_Orbit["apogee"] - rocket_eci.R0)/1000:.2f} m)')
    print(f'Final Trajectory Perigee (Target): {(apogee3_sol[-1] - rocket_eci.R0)/1000:.2f} m (Target: {(Target_Orbit["perigee"] - rocket_eci.R0)/1000:.2f} m)')
    print(f'Parking Orbit (Apogee, Perigee, Inclination): {(apogee3_sol[-1] - rocket_eci.R0)/1000:.2f} Km, {(perigee3_sol[-1] - rocket_eci.R0)/1000:.2f} Km, {np.rad2deg(i3_sol[-1]):.2f} deg')

    t1_vec = np.linspace(init_time, init_time+N1*dt1_sol, N1+1)
    t2_vec = np.linspace(t1_vec[-1], t1_vec[-1]+N2*dt2_sol, N2+1)
    t3_vec = np.linspace(t2_vec[-1], t2_vec[-1]+N3*dt3_sol, N3+1)
    # t3_vec = np.zeros((N3+1))
    # t3_vec[0] = t2_vec[-1]
    # for k in range(N3):
    #     t3_vec[k+1] = t3_vec[k] + dt3_sol*(1-0.5*k/(N3-1))
    
    # plot the results
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f'dt1={dt1_sol:.2f} s, dt2={dt2_sol:.2f} s, dt3={dt3_sol:.2f} s, LaunchAz={np.rad2deg(LaunchAz_sol):.2f} deg, PayloadMass={payload_mass_sol:.2f} kg', fontsize=16)
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.plot(x1_sol[0,:]/1000, x1_sol[1,:]/1000)
    ax_traj.set_title('Trajectory')
    ax_traj.set_ylabel('Altitude [Km]')
    ax_traj.set_xlabel('Distance [Km]')
    ax_traj.grid()
    ax_traj.set_aspect('equal', adjustable='box')

    ax_alt = fig.add_subplot(gs[0,1])
    ax_alt.plot(t1_vec, alt1_sol/1000, label='1st Stage')
    ax_alt.plot(t2_vec, alt2_sol/1000, label='2nd Stage before fairing separation')
    ax_alt.plot(t3_vec, alt3_sol/1000, label='2nd Stage after fairing separation')
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
    ax_vel.set_title('Velocity ECI')
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.set_xlabel('Time [s]')
    ax_vel.grid()
    ax_acc = fig.add_subplot(gs[1,2])
    ax_acc.plot(t1_vec[:-1], u1_sol[0,:])
    ax_acc.plot(t2_vec[:-1], np.linalg.norm(u2_sol, axis=0)/rocket_eci.SecondStage_Thrust)
    ax_acc.plot(t3_vec[:-1], np.linalg.norm(u3_sol, axis=0)/rocket_eci.SecondStage_Thrust)

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
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.FirstStage_EmptyMass)/1000, (payload_mass_sol+rocket_Booster.FirstStage_EmptyMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.LV_total_mass)/1000, (payload_mass_sol+rocket_Booster.LV_total_mass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.SecondStage_FullMass)/1000, (payload_mass_sol+rocket_Booster.SecondStage_FullMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.SecondStage_EmptyMass)/1000, (payload_mass_sol+rocket_Booster.SecondStage_EmptyMass)/1000], 'k--')
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
    ax_q.plot(t1_vec, Qdyn1_sol/1000.0)
    ax_q.set_title('Qynamic Pressure')
    ax_q.set_ylabel('Qynamic Pressure [KPa]')
    ax_q.grid()



    solution = {
        "u1": u1_sol,
        "x1": x1_sol,
        "dt1": dt1_sol,
        "u2": u2_sol,
        "x2": x2_sol,
        "dt2": dt2_sol,
        "dt3": dt3_sol,
        "x3": x3_sol,
        "u3": u3_sol,
        "payload_mass": payload_mass_sol,
        "LaunchAz": LaunchAz_sol,
        'Target_Orbit': Target_Orbit,
    }
    with open('solution.pickle', 'wb') as f:
        pickle.dump(solution, f)

    x1_N1_data = { "x1_N1": x1_sol[:, N1], "payload_mass": payload_mass_sol}
    with open('x1_N1_data.pickle', 'wb') as f:
        pickle.dump(x1_N1_data, f)



    plt.tight_layout()
    plt.show()