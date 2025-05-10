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
    rocket_return = LV_Type.BoosterReturn_2D(atmosphere=atmosphere)
    rocket_eci = LV_Type.LaunchVehicle_ECI()
    rocket_boostback = LV_Type.BoostBackBurn_2D()
    RecoveryStrategy = 'RTLS' # 'None', 'RTLS', 'ASDS'

    # define the optimization problem
    opti = ca.Opti()


# Phase 4 - Booster Return:
    # m0 = rocket_return.EmptyFirstStageMass + 20000.0
    N4 = 15
    x4_0 = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
    x4_before_reentry = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
    x4_after_reentry = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
    u4_reentry = opti.variable(rocket_return.nu, 1) # [T_factor]
    x4_before_landing = opti.variable(rocket_return.nx, 1) # [x, z, vx, vz, m]
    x4_landing = opti.variable(rocket_return.nx, N4+1) # [x, z, vx, vz, m]
    u4_landing = opti.variable(rocket_return.nu, N4) # [T_factor]
    dt4_reentry = opti.variable(1)
    dt4_ballistic = opti.variable(1)
    dt4_before_landing = opti.variable(1)
    dt4_landing = opti.variable(1)
    if RecoveryStrategy == 'RTLS':
        dt4_boostback = opti.variable(1)
        x4_boostback = opti.variable(rocket_boostback.nx, 1) # [x, z, vx, vz, m]
        u4_boostback = opti.variable(rocket_boostback.nu, 1) # [Tx, Tz]
    # Set the initial state
    with open("x1_N1_data.pickle", "rb") as f:
        x1_N1_data = pickle.load(f)

    # set the time step constraints
    if RecoveryStrategy == 'RTLS':
        opti.subject_to(dt4_boostback >= 1.0)
    opti.subject_to(dt4_ballistic >= 1.0)
    opti.subject_to(dt4_reentry >= 1.0)
    opti.subject_to(dt4_before_landing >= 1.0)
    opti.subject_to(dt4_landing*N4 >= 1.0)

    # dynamics
    opti.subject_to(x4_0[0:4,0] == x1_N1_data['x1_N1'][0:4])
    opti.subject_to(x4_0[4,0] == x1_N1_data['x1_N1'][4] - rocket_eci.SecondStage_FullMass - x1_N1_data['payload_mass'])
    if RecoveryStrategy == 'RTLS':
        opti.subject_to(x4_boostback == rocket_boostback.dynamics_kp1_M20(x4_0, u4_boostback, dt4_boostback))
        opti.subject_to(ca.norm_2(u4_boostback) <= 3.0/9.0)
        # opti.subject_to(x4_boostback[2] <= 0.0)

        opti.subject_to(x4_before_reentry == rocket_return.dynamics_kp1_M50(x4_boostback, 0, dt4_ballistic))

    else:
        opti.subject_to((x4_before_reentry-rocket_return.dynamics_kp1_M50(x4_0, 0, dt4_ballistic)) == 0.0)

    opti.subject_to(rocket_return.local_to_alt(x4_before_reentry)/60000.0 == 1.0)
    # Dynamics of entry burn
    opti.subject_to((x4_after_reentry - rocket_return.dynamics_kp1_M20(x4_before_reentry, u4_reentry, dt4_reentry)) == 0.0)
    # set the thrust constraints
    # opti.subject_to(u4_reentry*3.0 == 1.0)
    opti.subject_to(u4_reentry*3.0 <= 1.0)
    opti.subject_to(u4_reentry*3.0 >= rocket_return.FirstStage_MinThrust_Factor)

    # if RecoveryStrategy == 'RTLS':
        # opti.subject_to(ca.norm_2(ca.vertcat(x4_after_reentry[2], x4_after_reentry[3])) == 1000.0)
    # else:
    opti.subject_to(ca.norm_2(ca.vertcat(x4_after_reentry[2], x4_after_reentry[3])) <= 1400.0)
    opti.subject_to(rocket_return.local_to_alt(x4_after_reentry)/45000.0 == 1.0)

    # After Reentry, before landing burn
    opti.subject_to(x4_before_landing == rocket_return.dynamics_kp1_M20(x4_after_reentry, 0, dt4_before_landing))
    # opti.subject_to(rocket_return.local_to_alt(x4_before_landing)/1000.0 == 1.0)
    
    N_before_landing_constraints = 5
    x4_before_landing_intren = x4_after_reentry
    for i in range(N_before_landing_constraints):
        x4_before_landing_intren = rocket_return.dynamics_kp1(x4_before_landing_intren, 0, dt4_before_landing/N_before_landing_constraints)
        opti.subject_to(0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren)) * ca.norm_2(ca.vertcat(x4_before_landing_intren[2], x4_before_landing_intren[3]))**2 <= rocket_return.Booster_MaxDynamicPressure)
        # opti.subject_to(0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren)) * ca.norm_2(ca.vertcat(x4_before_landing_intren[2], x4_before_landing_intren[3]))**3 <= rocket_return.Booster_MaxHeatFlux)

    # Landing Burn:
    opti.subject_to(x4_landing[:,0] == x4_before_landing)
    for k in range(N4):
        opti.subject_to((x4_landing[:,k+1]-rocket_return.dynamics_kp1(x4_landing[:,k], u4_landing[:,k], dt4_landing)) == 0.0)
            
        opti.subject_to(u4_landing[k] <= 1.0/9.0)
        opti.subject_to(u4_landing[k] >= 0.9*rocket_return.EmptyFirstStageMass*rocket_return.g0/rocket_return.FirstStage_SL_Thrust)


    if RecoveryStrategy == 'RTLS':
        opti.subject_to(x4_landing[0,N4] == 0.0)
        opti.subject_to(x4_landing[1,N4] == 0.0)
    else:
        # zero altitude at the end of the flight
        opti.subject_to(rocket_return.local_to_alt(x4_landing[:,N4]) == 0.0) # landing altitude
    # minimal velocity at the end of the flight
    opti.subject_to(ca.norm_2(x4_landing[2:4,N4]) == 1.0) # zero velocity at the end of the flight
        
        # opti.subject_to(x4_landing[4,N4] >= rocket_return.EmptyFirstStageMass)
    # set the cost function
    cost = 0
    cost += -x4_landing[4,N4]
    cost += ca.sumsqr(u4_landing[1:] - u4_landing[:-1]) # minimize the thrust changes
    # cost += ca.sumsqr(u4_boostback[0,1:] - u4_boostback[0,:-1]) # minimize the thrust changes
    # cost += ca.sumsqr(u4_boostback[1,1:] - u4_boostback[1,:-1]) # minimize the thrust changes
    opti.minimize(cost)


    # set the solver
    opts = {
            "print_time": 0,  # Print timing, 
            "ipopt": {
            # "tol": 1e-8,  # Convergence tolerance
            "max_iter": 1000,  # Max iterations
            # "print_level": 0,  # Verbosity level
            # "alpha_for_y": "min",  # Fraction-to-boundary rule parameter
            # "timing_statistics": "yes", # Enable timing statistics
            # 'nlp_scaling_max_gradient': 100,
            # "obj_scaling_factor": 1e-1, # Scaling factor for the objective function
            # "mu_strategy": "adaptive",  # "monotone" or "adaptive" Strategy for updating the barrier parameter
            # "mu_init": 1e-11,  # 1e-1 Initial value of the barrier parameter
            # "mu_min": 1e-11,  # 1e-11 Smallest mu allowed — optimization stops when mu ≤ this
            # "mu_target": 0.0,  # 0.0 IPOPT drives mu towards this target; if 0, tries to reach exact KKT solution
            # "barrier_tol_factor": 5.0,  # Affects how mu and tolerances interact during convergence
        }}
    opti.solver('ipopt', opts)

    # calculate the initial guess for booster return
    x4_0_guess = np.zeros((rocket_return.nx, 1))
    x4_0_guess[0:4,0] = x1_N1_data['x1_N1'][0:4]
    x4_0_guess[4,0] = x1_N1_data['x1_N1'][4] - rocket_eci.SecondStage_FullMass - x1_N1_data['payload_mass']
    if RecoveryStrategy == 'RTLS':
        dt4_boostback_guess = 30.0
        x4_boostback_guess = np.zeros((rocket_boostback.nx, 1))
        u4_boostback_guess = np.zeros((rocket_boostback.nu, 1))
        iter = 0
        while iter < 1:
            u4_boostback_guess[0] = -3.0/9.0 * np.cos(np.deg2rad(-10.0))
            u4_boostback_guess[1] = -3.0/9.0 * np.sin(np.deg2rad(-10.0))
            x4_boostback_guess = rocket_boostback.dynamics_kp1_M20(x4_0_guess, u4_boostback_guess, dt4_boostback_guess).full().flatten()
            alt = rocket_Booster.local_to_alt(x4_boostback_guess)
            vel = np.linalg.norm(x4_boostback_guess[2:4])
            t_bal = (-x4_boostback_guess[3]  + np.sqrt(x4_boostback_guess[3]**2 +2*rocket_boostback.g0*alt))/(0.5*rocket_boostback.g0)
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
            x4_before_reentry_guess = rocket_return.dynamics_kp1_M50(x4_boostback_guess, 0, dt4_ballistic_guess)
        else:
            x4_before_reentry_guess = rocket_return.dynamics_kp1_M50(x4_0_guess, 0, dt4_ballistic_guess)
        alt = ca.norm_2(ca.vertcat(x4_before_reentry_guess[0,0], rocket_return.R0+x4_before_reentry_guess[1,0])) - rocket_return.R0
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
        x4_after_reentry_guess = rocket_return.dynamics_kp1_M20(x4_before_reentry_guess, u4_reentry_guess, dt4_reentry_guess)
        alt = ca.norm_2(ca.vertcat(x4_after_reentry_guess[0,0], rocket_return.R0+x4_after_reentry_guess[1,0])) - rocket_return.R0
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
        x4_before_landing_guess = rocket_return.dynamics_kp1_M20(x4_after_reentry_guess, 0, dt4_before_landing_guess)
        alt = ca.norm_2(ca.vertcat(x4_before_landing_guess[0,0], rocket_return.R0+x4_before_landing_guess[1,0])) - rocket_return.R0
        vel = ca.norm_2(ca.vertcat(x4_before_landing_guess[2,0], x4_before_landing_guess[3,0]))
        if abs(alt - alt4_before_landing) > 10:
            dt4_before_landing_guess += 0.9*(alt - alt4_before_landing)/vel
            iter += 1
        else:
            break

    # initial guess for the landing burn
    vel = np.linalg.norm(x4_before_landing_guess[2:4])
    alt  = rocket_return.local_to_alt(x4_before_landing_guess)
    dt4_landing_guess = 0.5*vel**2/alt / N4
    x4_landing_guess = np.zeros((rocket_return.nx, N4+1))
    u4_landing_guess = np.zeros((rocket_return.nu, N4))
    iter = 0
    thrust_factor = 1.0
    while iter < 20:
        u4_landing_guess[0,0] = 0.1111
        x4_landing_guess[:,0] = x4_before_landing_guess.full().flatten()
        for k in range(N4):
            vel = np.linalg.norm(x4_landing_guess[2:4,k])
            alt = rocket_return.local_to_alt(x4_landing_guess[:,k])
            # if vel > 75.0:
            #     u4_landing_guess[0,k] = 0.1111 * thrust_factor
            # else:
            #     u4_landing_guess[0,k] = x4_landing_guess[4,k] * rocket_return.g0/rocket_return.FirstStage_SL_Thrust
            g_vec = -np.array([x4_landing_guess[0,k], rocket_return.R0 + x4_landing_guess[1,k]])
            g_vec /= np.linalg.norm(g_vec)
            g = rocket_return.g0 * (rocket_return.R0/(rocket_return.R0 + x4_landing_guess[1,k]))**2
            v_g = np.dot(g_vec, x4_landing_guess[2:4,k])
            u4_landing_guess[0,k] = np.dot(g_vec, x4_landing_guess[2:4,k])/vel * x4_landing_guess[4,k] * g/rocket_return.FirstStage_SL_Thrust
            if vel > 1.0:
                u4_landing_guess[0,k] += (v_g**2)/(2*alt)*x4_landing_guess[4,k]/rocket_return.FirstStage_SL_Thrust
            x4_landing_guess[:,k+1] = rocket_return.dynamics_kp1(x4_landing_guess[:,k], u4_landing_guess[0,k], dt4_landing_guess).full().flatten()
            alt = rocket_return.local_to_alt(x4_landing_guess[:,k+1])
            vel = np.linalg.norm(x4_landing_guess[2:4,k+1])
            if alt < -1.0:
                break
        if alt > 20:
            dt4_landing_guess += 0.2*(alt)/vel/N4
            iter += 1
        elif abs(vel) > 20:
            thrust_factor += 0.5*vel/10
            iter += 1
        elif alt < -1.0:
            dt4_landing_guess *= k/N4
            iter += 1
        else:
            break
    t4_landing_guess = np.array(dt4_ballistic_guess + dt4_reentry_guess + dt4_before_landing_guess) + np.linspace(0, dt4_landing_guess*N4, N4+1)
    t4_landing_guess = t4_landing_guess.reshape((-1,))

    
    # Ballistic Phase
    opti.set_initial(x4_0, x4_0_guess)
    opti.set_initial(dt4_ballistic, dt4_ballistic_guess)
    opti.set_initial(x4_before_reentry, x4_before_reentry_guess)
    opti.set_initial(x4_after_reentry, x4_after_reentry_guess)
    opti.set_initial(u4_reentry, u4_reentry_guess)
    opti.set_initial(dt4_reentry, dt4_reentry_guess)
    opti.set_initial(dt4_before_landing, dt4_before_landing_guess)
    opti.set_initial(x4_before_landing, x4_before_landing_guess)
    opti.set_initial(dt4_landing, dt4_landing_guess)
    opti.set_initial(x4_landing, x4_landing_guess)
    opti.set_initial(u4_landing, u4_landing_guess)
    if RecoveryStrategy == 'RTLS':
        opti.set_initial(dt4_boostback, dt4_boostback_guess)
        opti.set_initial(x4_boostback, x4_boostback_guess)
        opti.set_initial(u4_boostback, u4_boostback_guess)

    dt4_boostback_sol = 0.0
    x4_boostback_sol = np.zeros((rocket_boostback.nx, 1))
    u4_boostback_sol = np.zeros((rocket_boostback.nu, 1))
    # Solve the optimization problem
    try:
        sol = opti.solve()

        x4_0_sol = np.array(sol.value(x4_0))
        dt4_ballistic_sol = np.array(sol.value(dt4_ballistic))
        x4_before_reentry_sol = np.array(sol.value(x4_before_reentry))
        x4_after_reentry_sol = np.array(sol.value(x4_after_reentry))
        u4_reentry_sol = np.array(sol.value(u4_reentry))
        dt4_reentry_sol = np.array(sol.value(dt4_reentry))
        dt4_before_landing_sol = np.array(sol.value(dt4_before_landing))
        x4_before_landing_sol = np.array(sol.value(x4_before_landing))
        dt4_landing_sol = np.array(sol.value(dt4_landing))
        x4_landing_sol = np.array(sol.value(x4_landing))
        u4_landing_sol = np.array(sol.value(u4_landing))
        if RecoveryStrategy == 'RTLS':
            dt4_boostback_sol = np.array(sol.value(dt4_boostback))
            x4_boostback_sol = np.array(sol.value(x4_boostback))
            u4_boostback_sol = np.array(sol.value(u4_boostback))


    except:
        x4_0_sol = opti.debug.value(x4_0)
        dt4_ballistic_sol = opti.debug.value(dt4_ballistic)
        x4_before_reentry_sol = opti.debug.value(x4_before_reentry)
        x4_after_reentry_sol = opti.debug.value(x4_after_reentry)
        u4_reentry_sol = opti.debug.value(u4_reentry)
        dt4_reentry_sol = opti.debug.value(dt4_reentry)
        dt4_before_landing_sol = opti.debug.value(dt4_before_landing)
        x4_before_landing_sol = opti.debug.value(x4_before_landing)
        dt4_landing_sol = opti.debug.value(dt4_landing)
        x4_landing_sol = opti.debug.value(x4_landing)
        u4_landing_sol = opti.debug.value(u4_landing)
        if RecoveryStrategy == 'RTLS':
            dt4_boostback_sol = opti.debug.value(dt4_boostback)
            x4_boostback_sol = opti.debug.value(x4_boostback)
            u4_boostback_sol = opti.debug.value(u4_boostback)


    # grad_f = ca.gradient(opti.f, opti.x)
    # grad_func = ca.Function("grad_f", [opti.x], [grad_f])
    # g_val = grad_func(opti.debug.value(opti.x))
    # print("Gradient of f(x):", g_val.full().flatten())


    return_data = {
        'x4_0': x4_0_sol,
        'dt4_ballistic': dt4_ballistic_sol,
        'x4_before_reentry': x4_before_reentry_sol,
        'x4_after_reentry': x4_after_reentry_sol,
        'u4_reentry': u4_reentry_sol,
        'dt4_reentry': dt4_reentry_sol,
        'dt4_before_landing': dt4_before_landing_sol,
        'x4_before_landing': x4_before_landing_sol,
        'dt4_landing': dt4_landing_sol,
        'x4_landing': x4_landing_sol,
        'u4_landing': u4_landing_sol,
        'dt4_boostback': dt4_boostback_sol,
        'x4_boostback': x4_boostback_sol,
        'u4_boostback': u4_boostback_sol,
    }
    with open("ReturnSolution.pickle", "wb") as f:
        pickle.dump(return_data, f)

    alt4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[0], rocket_return.R0+x4_0_sol[1])) - rocket_return.R0
    vel4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[2], x4_0_sol[3]))

    if RecoveryStrategy == 'RTLS':
        t4_boostback_vec = np.linspace(0, dt4_boostback_sol, N4+1)
        alt4_boostback_sol = np.zeros((N4+1,1))
        vel4_boostback_sol = np.zeros((N4+1,1))
        x4_boostback_vec_sol = np.zeros((rocket_boostback.nx, N4+1))
        x4_boostback_vec_sol[:,0] = x4_0_sol
        for k in range(N4+1):
            if k<N4:
                x4_boostback_vec_sol[:,k+1] = rocket_boostback.dynamics_kp1(x4_boostback_vec_sol[:,k], u4_boostback_sol, dt4_boostback_sol/N4).full().flatten()
            alt4_boostback_sol[k] = ca.norm_2(ca.vertcat(x4_boostback_vec_sol[0,k], rocket_return.R0+x4_boostback_vec_sol[1,k])) - rocket_return.R0
            vel4_boostback_sol[k] = ca.norm_2(ca.vertcat(x4_boostback_vec_sol[2,k], x4_boostback_vec_sol[3,k]))

    N4_ballistic = 50
    x4_ballistic_sol = np.zeros((rocket_return.nx, N4_ballistic))
    alt4_ballistic_sol, vel4_ballistic_sol = np.zeros((N4_ballistic,1)), np.zeros((N4_ballistic,1))
    Qdyn4_ballistic_sol = np.zeros((N4_ballistic,1))
    t4_ballistic_vec = np.zeros((N4_ballistic,1))
    if RecoveryStrategy == 'RTLS':
        x4_ballistic_sol[:,0] = x4_boostback_sol.reshape((-1,))
    else:
        x4_ballistic_sol[:,0] = x4_0_sol
    t4_ballistic_vec[0] = 0.0 if RecoveryStrategy != 'RTLS' else dt4_boostback_sol
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
    x4_reentry_sol[:,0] = x4_ballistic_sol[:,-1]
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
        t4_before_landing_vec[k] = t4_before_landing_vec[k-1] + dt4_before_landing_sol/(N4_before_landing-1)
        x4_before_landing_vec_sol[:,k] = rocket_return.dynamics_kp1(x4_before_landing_vec_sol[:,k-1], 0, dt4_before_landing_sol/(N4_before_landing-1)).full().flatten()
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
    
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax_traj = fig.add_subplot(gs[0, 0])    
    # plot the results
    if RecoveryStrategy != 'RTLS':
        x_earth = np.linspace(0, 800000,1000)
        z_earth = np.sqrt(rocket_return.R0**2 - x_earth**2) - rocket_return.R0
        ax_traj.plot(x_earth/1000, z_earth/1000, '--k', label='Earth')
    else:
        ax_traj.plot(x4_boostback_vec_sol[0,:]/1000, x4_boostback_vec_sol[1,:]/1000, label='Boostback Phase')
        ax_traj.plot(x4_boostback_sol[0]/1000, x4_boostback_sol[1]/1000, 's', label='Booster Return')


    ax_traj.plot(x4_ballistic_sol[0,:]/1000, x4_ballistic_sol[1,:]/1000, label='Ballistic Phase')
    ax_traj.plot(x4_before_reentry_sol[0]/1000, x4_before_reentry_sol[1]/1000,'s', label='Booster Return')
    ax_traj.plot(x4_reentry_sol[0,:]/1000, x4_reentry_sol[1,:]/1000, label='Reentry Phase')
    ax_traj.plot(x4_before_landing_vec_sol[0,:]/1000, x4_before_landing_vec_sol[1,:]/1000, label='Before Landing Phase')
    ax_traj.plot(x4_landing_sol[0,:]/1000, x4_landing_sol[1,:]/1000,'-x', label='Landing Phase')
    ax_traj.set_title('Trajectory')
    ax_traj.set_ylabel('Altitude [Km]')
    ax_traj.set_xlabel('Distance [Km]')
    ax_traj.grid()
    ax_traj.set_aspect('equal', adjustable='box')

    ax_alt = fig.add_subplot(gs[0,1])
    if RecoveryStrategy == 'RTLS':
        ax_alt.plot(t4_boostback_vec, alt4_boostback_sol/1000, label='Boostback')
    ax_alt.plot(t4_ballistic_vec, alt4_ballistic_sol/1000, label='Ballistic')
    ax_alt.plot(t4_ballistic_vec[-1], alt4_before_reentry_sol/1000,'s', label='Booster Return')
    ax_alt.plot(t4_reentry_vec, alt4_reentry_sol/1000, label='Reentry')
    ax_alt.plot(t4_before_landing_vec, alt4_before_landing_sol/1000, label='Before Landing')
    ax_alt.plot(t4_landing_vec, alt4_landing_sol/1000,'-x', label='Landing')
    ax_alt.set_title('Altitude')
    ax_alt.set_ylabel('Altitude [Km]')
    ax_alt.set_xlabel('Time [s]')
    ax_alt.grid()

    ax_vel = fig.add_subplot(gs[0,2])
    if RecoveryStrategy == 'RTLS':
        ax_vel.plot(t4_boostback_vec, vel4_boostback_sol, label='Boostback')
    ax_vel.plot(t4_ballistic_vec, vel4_ballistic_sol, label='Booster Return')
    ax_vel.plot(t4_ballistic_vec[-1], vel4_before_reentry_sol,'s', label='Booster Return')
    ax_vel.plot(t4_reentry_vec, vel4_reentry_sol, label='Reentry')
    ax_vel.plot(t4_before_landing_vec, vel4_before_landing_sol, label='Before Landing')
    ax_vel.plot(t4_landing_vec, vel4_landing_sol,'-x', label='Landing')
    ax_vel.set_title('Velocity ECI')
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.set_xlabel('Time [s]')
    ax_vel.grid()
    ax_acc = fig.add_subplot(gs[1,2])
    ax_acc.plot(t4_ballistic_vec, t4_ballistic_vec*0)
    ax_acc.plot(t4_ballistic_vec[-1], t4_ballistic_vec[-1]*0)
    ax_acc.plot(t4_reentry_vec, u4_reentry_sol * np.ones_like(t4_reentry_vec), label='Booster Return')
    ax_acc.plot(t4_before_landing_vec, t4_before_landing_vec*0, label='Before Landing')
    ax_acc.plot(t4_landing_vec[:-1], u4_landing_sol,'-x', label='Landing')
    # ax_acc.plot(t4_landing_guess[:-1], u4_landing_guess.T,'-x', label='Landing guess')
    ax_acc.set_title('Thrust Factor')
    ax_acc.set_ylabel('Thrust Factor [-]')
    ax_acc.set_xlabel('Time [s]')
    ax_acc.grid()
    # ax_acc.set_ylim(0, 1.1)
    # ax_alpha = fig.add_subplot(gs[2,2])
    # ax_alpha.plot(t4_vec[:-1], np.rad2deg(u4_sol[1,:]))
    # ax_alpha.set_title('Angle of Attack')
    # ax_alpha.set_ylabel('Angle of Attack [deg]')
    # ax_alpha.set_xlabel('Time [s]')
    # ax_alpha.grid()
    ax_mass = fig.add_subplot(gs[2,0])
    if RecoveryStrategy == 'RTLS':
        ax_mass.plot(t4_boostback_vec, x4_boostback_vec_sol[4,:]/1000, label='Boostback')
        ax_mass.plot(dt4_boostback_sol, x4_boostback_sol[4]/1000,'s', label='Booster Return')
    ax_mass.plot(t4_ballistic_vec, x4_ballistic_sol[4,:]/1000)
    ax_mass.plot(0.0, x4_0_sol[4]/1000,'s', label='Booster Return')
    ax_mass.plot(t4_reentry_vec, x4_reentry_sol[4,:]/1000)
    ax_mass.plot(t4_before_landing_vec, x4_before_landing_vec_sol[4,:]/1000)
    ax_mass.plot(t4_landing_vec, x4_landing_sol[4,:]/1000,'-x', label='Landing')
    ax_mass.plot([0, t4_landing_vec[-1,0]], [rocket_return.EmptyFirstStageMass/1000, rocket_return.EmptyFirstStageMass/1000], 'r--', label='Empty Mass')
    ax_mass.set_title('Mass')
    ax_mass.set_ylabel('Mass [Tons]')
    ax_mass.set_xlabel('Time [s]')
    ax_mass.grid()


            

    ax_q = fig.add_subplot(gs[2,1])
    ax_q.plot(t4_ballistic_vec, Qdyn4_ballistic_sol/1000.0, label='Booster Return')
    ax_q.plot(t4_ballistic_vec[-1], Qdyn4_before_reentry_sol[0]/1000.0,'s', label='Booster Return')
    ax_q.plot(t4_reentry_vec, Qdyn4_reentry_sol/1000.0, label='Reentry')
    ax_q.plot(t4_before_landing_vec, Qdyn4_before_landing_sol/1000.0, label='Before Landing')
    ax_q.plot(t4_landing_vec, Qdyn4_landing_sol/1000.0,'-x', label='Landing')
    x4_before_landing_intren_sol = x4_reentry_sol[:,-1]
    for i in range(N_before_landing_constraints):
        x4_before_landing_intren_sol = rocket_return.dynamics_kp1(x4_before_landing_intren_sol, 0, dt4_before_landing_sol/N_before_landing_constraints)
        Qdyn = 0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren_sol)) * (x4_before_landing_intren_sol[2]**2 + x4_before_landing_intren_sol[3]**2)
        ax_q.plot(t4_reentry_vec[-1] + (i+1)*dt4_before_landing_sol/N_before_landing_constraints, Qdyn/1000.0, 'ro')
    ax_q.set_title('Qynamic Pressure')
    ax_q.set_ylabel('Qynamic Pressure [KPa]')
    ax_q.grid()




    plt.tight_layout()
    plt.show()
        




    