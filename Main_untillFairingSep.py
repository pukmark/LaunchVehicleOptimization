#!/usr/bin/env python3

import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.gridspec as gridspec
import casadi as ca

import Atmosphere_Type as Atm_Type
import LV_Type as LV_Type




if __name__ == '__main__':
    # Create an instance of the AtmosphereType class
    atmosphere = Atm_Type.AtmosphereType()
    # Create an instance of the VLType class
    rocket_Booster = LV_Type.BoosterLaunchVehicle_2D(atmosphere=atmosphere)

    # Target orbit parameters
    Target_Orbit = {"a": rocket_Booster.R0 + 150.0*1000.0, # semi-major axis
                    "e": 0.0, # eccentricity
                    "i": 0.0, # inclination
                    }
    Target_Orbit["b"] = Target_Orbit["a"] * np.sqrt(1.0 - Target_Orbit["e"]**2) # semi-minor axis
    Target_Orbit["apogee"] = Target_Orbit["a"] * (1.0 + Target_Orbit["e"]) # apogee
    Target_Orbit["perigee"] = Target_Orbit["a"] * (1.0 - Target_Orbit["e"]) # perigee

    # define the initial state - optimization starts at 1 sec after lift off
    init_time = 5.0
    init_acc = rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.LV_total_mass) - rocket_Booster.g0 * (rocket_Booster.R0 / (rocket_Booster.R0 + rocket_Booster.LaunchAltitude))**2
    init_vz = init_acc * init_time
    init_vx = 0.0
    init_z = 0.5 * init_acc * init_time**2
    init_x = 0.0
    init_m = rocket_Booster.LV_total_mass - (rocket_Booster.FirstStage_SL_Thrust / (rocket_Booster.FirstStage_SL_Isp * rocket_Booster.g0)) * init_time
    x0 = np.array([init_x, init_z, init_vx, init_vz])

    # define the optimization problem
    opti = ca.Opti()

    # First Phase - Booster 
    N1 = 40
    payload_mass = opti.variable(1)
    x1 = opti.variable(rocket_Booster.nx, N1+1) # [x, z, vx, vz, m]
    u1 = opti.variable(rocket_Booster.nu, N1) # [T_factor, alpha]
    dt1 = opti.variable(1)
    LaunchAz = opti.variable(1)

    # Set the initial state
    opti.subject_to(x1[:4,0] == x0)
    opti.subject_to(x1[4,0] == rocket_Booster.LV_total_mass + payload_mass)
    opti.subject_to(payload_mass >= 1000.0)
    # set the constraints:
    for k in range(N1):
        # set the dynamics
        opti.subject_to(x1[:,k+1] == rocket_Booster.dynamics_kp1(x1[:,k], u1[:,k], dt1))
        # set the thrust constraints
        opti.subject_to(u1[0,k] <= 1.0)
        opti.subject_to(u1[0,k] >= rocket_Booster.FirstStage_MinThrust_Factor)
        opti.subject_to(u1[1,k] <= rocket_Booster.FirstStage_MaxAlpha)
        opti.subject_to(u1[1,k] >= -rocket_Booster.FirstStage_MaxAlpha)
        # set the time step constraints
        opti.subject_to(dt1 >= 0.01)
        opti.subject_to(dt1 <= 5.0)
        # set the state constraints
        Alt = ca.norm_2(ca.vertcat(x1[0,k], rocket_Booster.R0 + x1[1,k])) - rocket_Booster.R0
        opti.subject_to(Alt >= 0.0)
        opti.subject_to(0.5*atmosphere.rho_fun(Alt)*(x1[2,k]**2+x1[3,k]**2)<=25000.0)
        opti.subject_to(u1[1,k]*0.5*atmosphere.rho_fun(Alt)*(x1[2,k]**2+x1[3,k]**2)<=1000.0)
        opti.subject_to(-u1[1,k]*0.5*atmosphere.rho_fun(Alt)*(x1[2,k]**2+x1[3,k]**2)<=1000.0)
    
    # set the final state constraints
    opti.subject_to(x1[4,N1] >= rocket_Booster.FirstStage_EmptyMass + payload_mass)
    opti.subject_to(x1[1,N1] >= 60000.0)

    # Phase 2 - Second Stage, before fairing separation (if needed)
    rocket_eci = LV_Type.LaunchVehicle_ECI()
    if rocket_eci.FairingMass > 0.0:
        N2 = 10
        x2 = opti.variable(rocket_eci.nx, N2+1) # [x, y, z, vx, vy, vz, m]
        u2 = opti.variable(rocket_eci.nu, N2) # [fx, fy, fz]
        dt2 = opti.variable(1)

        # Set the initial state - the final state of the boost phase transformed to ECI
        opti.subject_to(LaunchAz >= -0.9*np.pi)
        opti.subject_to(LaunchAz <= 0.9*np.pi)
        eci_pos, eci_vel = rocket_eci.local_to_eci_func(ca.vertcat(x1[0,N1], 0.0, x1[1,N1]), ca.vertcat(x1[2,N1], 0.0, x1[3,N1]), LaunchAz)
        opti.subject_to(x2[0:3,0] == eci_pos)
        opti.subject_to(x2[3:6,0] == eci_vel)
        opti.subject_to(x2[6,0] == rocket_eci.SecondStage_FullMass + payload_mass)
        for k in range(N2):
            # set the dynamics
            opti.subject_to(x2[:,k+1] == rocket_eci.dynamics_kp1(x2[:,k], u2[:,k], dt2))
            # set the thrust constraints
            opti.subject_to(ca.norm_2(u2[:,k]) == rocket_eci.SecondStage_Thrust)
            # set the time step constraints
            opti.subject_to(dt2 >= 0.01)
            opti.subject_to(dt2 <= 6.0)

        # set the final state constraints
        Alt_N2 = ca.norm_2(x2[0:3,N2]) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
        opti.subject_to(Alt_N2 == rocket_eci.FairingSeparationAltitude)

            
    # set the cost function
    cost = 0.5*(ca.sumsqr(u1[0,:]) + ca.sumsqr(u1[1,:])/rocket_Booster.FirstStage_MaxAlpha**2)
    cost += -payload_mass - 0.5*ca.norm_2(x2[3:6,N2])**2
    opti.minimize(cost)


    # set the solver
    opts = {
            # "print_time": 1,  # Print timing, 
            "ipopt": {
            "tol": 1e-4,  # Convergence tolerance
            "max_iter": 1000,  # Max iterations
            # "print_level": 0,  # Verbosity level
            # "alpha_for_y": "min",  # Fraction-to-boundary rule parameter
            # "timing_statistics": "yes", # Enable timing statistics
            "nlp_scaling_method": "None", # Scaling method
        }}
    opti.solver('ipopt', opts)



    # calculate the initial guess for first stage 
    x1_guess = np.zeros((rocket_Booster.nx, N1+1))
    u1_guess = np.zeros((rocket_Booster.nu, N1))
    dt_guess = 2.6
    payload_mass_guess = 10000.0

    x1_guess[:-1,0] = x0
    x1_guess[-1,0] = rocket_Booster.LV_total_mass+payload_mass_guess
    for k in range(N1):
        if x1_guess[1,k]<5000 or x1_guess[1,k]>15000:
            u1_guess[0,k] = 1.0
        else:
            u1_guess[0,k] = 0.50
        if np.atan2(x1_guess[3,k], x1_guess[2,k]) > 80.0*np.pi/180.0 and np.linalg.norm(x1_guess[2:4,k]) > 100.0:
            u1_guess[1,k] = -0.01
        else:
            u1_guess[1,k] = 0.0
        
        Alt = ca.norm_2(ca.vertcat(x1_guess[0,k], rocket_Booster.R0 + x1_guess[1,k])) - rocket_Booster.R0
        Q = 0.5 * (x1_guess[2,k]**2 + x1_guess[3,k]**2) * atmosphere.rho_fun(Alt)
        x1_guess[:,k+1] = rocket_Booster.dynamics_kp1(x1_guess[:,k], u1_guess[:,k], dt_guess).full().flatten()

    Q_guess = np.zeros((N1+1))
    for k in range(N1+1):
        Alt = ca.norm_2(ca.vertcat(x1_guess[0,k], rocket_Booster.R0 + x1_guess[1,k])) - rocket_Booster.R0
        Q_guess[k] = 0.5 * (x1_guess[2,k]**2 + x1_guess[3,k]**2) * atmosphere.rho_fun(Alt)

    opti.set_initial(payload_mass, payload_mass_guess)
    opti.set_initial(dt1, dt_guess)
    for k in range(N1+1):
        opti.set_initial(x1[:,k], x1_guess[:,k])
    for k in range(N1):
        opti.set_initial(u1[:,k], u1_guess[:,k])
    

    # calculate the initial guess for second stage before fairing separation, if needed
    if rocket_eci.FairingMass > 0.0:
        x2_guess = np.zeros((rocket_eci.nx, N2+1))
        u2_guess = np.zeros((rocket_eci.nu, N2))
        pos, vel = rocket_eci.local_to_eci_func(ca.vertcat(x1_guess[0,N1], 0.0, x1_guess[1,N1]), ca.vertcat(x1_guess[2,N1], 0.0, x1_guess[3,N1]), np.pi/2.0)
        x2_guess[0:3,0] = pos.full().flatten()
        x2_guess[3:6,0] = vel.full().flatten()
        x2_guess[6,0] = rocket_eci.SecondStage_FullMass + payload_mass_guess
        for k in range(N2):
            u2_guess[0:3,k] = (x2_guess[3:6,k] / ca.norm_2(x2_guess[3:6,k])).full().flatten() * rocket_eci.SecondStage_Thrust
            x2_guess[:,k+1] = rocket_eci.dynamics_kp1(x2_guess[:,k], u2_guess[:,k], dt_guess).full().flatten()
            alt = ca.norm_2(x2_guess[0:3,k+1]) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)

        opti.set_initial(LaunchAz, np.pi/2.0)
        opti.set_initial(dt2, dt_guess)
        for k in range(N2+1):
            opti.set_initial(x2[:,k], x2_guess[:,k])
        for k in range(N2):
            opti.set_initial(u2[:,k], u2_guess[:,k])

    # Solve the optimization problem
    sol = opti.solve()

    u1_sol = np.array(sol.value(u1))
    x1_sol = np.array(sol.value(x1))
    dt1_sol = np.array(sol.value(dt1))
    u2_sol = np.array(sol.value(u2))
    x2_sol = np.array(sol.value(x2))
    dt2_sol = np.array(sol.value(dt2))
    payload_mass_sol = np.array(sol.value(payload_mass))
    LaunchAz_sol = np.array(sol.value(LaunchAz))

    Isp1_sol = np.zeros((N1))
    Qdyn1_sol = np.zeros((N1+1))
    a1_sol, b1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    apogee1_sol, perigee1_sol = np.zeros((N1+1)), np.zeros((N1+1))
    alt1_sol = np.zeros((N1+1))
    for k in range(N1+1):
        if k < N1:
            Isp1_sol[k] = rocket_Booster.ISP_calc(x1_sol[:,k], u1_sol[:,k]).full().flatten()[0]
        Qdyn1_sol[k] = 0.5 * (x1_sol[2,k]**2 + x1_sol[3,k]**2) * atmosphere.rho_fun(x1_sol[1,k])
        pos, vel = rocket_eci.local_to_eci_func(ca.vertcat(x1_sol[0,k], 0.0, x1_sol[1,k]), ca.vertcat(x1_sol[2,k], 0.0, x1_sol[3,k]), LaunchAz_sol)
        alt1_sol[k] = np.linalg.norm(pos) - rocket_eci.R0 / np.sqrt(1.0 - rocket_eci.e**2 * np.sin(np.deg2rad(rocket_eci.LaunchLatitude))**2)
        h1 = ca.cross(pos, vel)
        e1 = ca.cross(vel, h1) / (rocket_eci.mu) - pos / np.linalg.norm(pos)
        eps1 = 0.5*ca.sumsqr(vel) - rocket_eci.mu / np.linalg.norm(pos)
        a1_sol[k] = -rocket_eci.mu / (2.0*eps1)
        b1_sol[k] = a1_sol[k] * np.sqrt(1.0 - ca.sumsqr(e1))
        apogee1_sol[k] = a1_sol[k] * (1.0 + np.linalg.norm(e1))
        perigee1_sol[k] = a1_sol[k] * (1.0 - np.linalg.norm(e1))

    Isp2_sol = np.zeros((N2))
    Alpha2_sol = np.zeros((N2))
    a2_sol, b2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    apogee2_sol, perigee2_sol = np.zeros((N2+1)), np.zeros((N2+1))
    alt2_sol = np.zeros((N2+1))
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


    t1_vec = np.linspace(init_time, init_time+N1*dt1_sol, N1+1)
    t2_vec = np.linspace(t1_vec[-1], t1_vec[-1]+N2*dt2_sol, N2+1)
    # plot the results
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f'First Stage Trajectory, dt1={dt1_sol:.2f} s, dt2={dt2_sol:.2f} s, LaunchAz={np.rad2deg(LaunchAz_sol):.2f} deg, PayloadMass={payload_mass_sol:.2f} kg', fontsize=16)
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax_traj = fig.add_subplot(gs[0:2, 0])
    ax_traj.plot(x1_sol[0,:]/1000, x1_sol[1,:]/1000)
    ax_traj.set_title('Trajectory')
    ax_traj.set_ylabel('Altitude [Km]')
    ax_traj.set_xlabel('Distance [Km]')
    ax_traj.grid()
    ax_traj.set_aspect('equal', adjustable='box')

    ax_alt = fig.add_subplot(gs[0,1])
    ax_alt.plot(t1_vec, alt1_sol, label='1st Stage')
    ax_alt.plot(t2_vec, alt2_sol, label='2nd Stage')
    ax_alt.set_title('Altitude')
    ax_alt.set_ylabel('Altitude [Km]')
    ax_alt.set_xlabel('Time [s]')
    ax_alt.grid()

    ax_vel = fig.add_subplot(gs[0,2])
    ax_vel.plot(t1_vec, np.linalg.norm(x1_sol[2:4,:], axis=0), label='1st Stage')
    ax_vel.plot(t2_vec, np.linalg.norm(x2_sol[3:6,:], axis=0), label='2nd Stage')
    ax_vel.set_title('Velocity')
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.set_xlabel('Time [s]')
    ax_vel.grid()
    ax_acc = fig.add_subplot(gs[1,2])
    ax_acc.plot(t1_vec[:-1], u1_sol[0,:])
    ax_acc.plot(t2_vec[:-1], np.ones((N2)))
    ax_acc.set_title('Thrust Factor')
    ax_acc.set_ylabel('Thrust Factor [-]')
    ax_acc.set_xlabel('Time [s]')
    ax_acc.grid()
    ax_acc.set_ylim(0, 1.1)
    ax_alpha = fig.add_subplot(gs[2,2])
    ax_alpha.plot(t1_vec[:-1], np.rad2deg(u1_sol[1,:]))
    ax_alpha.plot(t2_vec[:-1], np.rad2deg(Alpha2_sol))
    ax_alpha.set_title('Angle of Attack')
    ax_alpha.set_ylabel('Angle of Attack [deg]')
    ax_alpha.set_xlabel('Time [s]')
    ax_alpha.grid()
    ax_mass = fig.add_subplot(gs[2,0])
    ax_mass.plot(t1_vec, x1_sol[4,:]/1000)
    ax_mass.plot(t2_vec, x2_sol[6,:]/1000)
    ax_mass.plot([0, t1_vec[-1]], [rocket_Booster.FirstStage_EmptyMass/1000, rocket_Booster.FirstStage_EmptyMass/1000], 'k--')
    ax_mass.plot([0, t1_vec[-1]], [rocket_Booster.LV_total_mass/1000, rocket_Booster.LV_total_mass/1000], 'k--')
    ax_mass.plot([0, t1_vec[-1]], [rocket_Booster.SecondStage_FullMass/1000, rocket_Booster.SecondStage_FullMass/1000], 'k--')
    ax_mass.plot([0, t1_vec[-1]], [rocket_Booster.SecondStage_EmptyMass/1000, rocket_Booster.SecondStage_EmptyMass/1000], 'k--')
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




    plt.tight_layout()
    plt.show()
        




    