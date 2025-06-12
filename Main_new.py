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
import LV_Optimization_Type as LV_Opt




if __name__ == '__main__':
    # Create an instance of the AtmosphereType class
    atmosphere = Atm_Type.AtmosphereType()
    # Create an instance of the VLType class
    rocket_Booster = LV_Type.BoosterLaunchVehicle_2D(atmosphere=atmosphere)
    rocket_eci = LV_Type.LaunchVehicle_ECI()
    rocket_return = LV_Type.BoosterReturn_2D(atmosphere=atmosphere)
    rocket_boostback = LV_Type.BoostBackBurn_2D()
    
    # Target orbit parameters for LEO
    Target_Orbit = {"apogee": rocket_Booster.R0 + 400.0*1000.0, # semi-major axis
                    "perigee": rocket_Booster.R0 + 200.0*1000.0, # semi-minor axis
                    "i": np.deg2rad(53.3), # inclination [rad]
                    }

    # Target orbit parameters for GTO
    # Target_Orbit = {"apogee": rocket_Booster.R0 + 35786.0*1000.0, # semi-major axis
    #                 "perigee": rocket_Booster.R0 + 200.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(rocket_Booster.LaunchLatitude), # inclination [rad]
    #                 }

    # # Target orbit parameters for SSO
    # Target_Orbit = {"apogee": rocket_Booster.R0 + 780 * 1000.0, # semi-major axis
    #                 "perigee": rocket_Booster.R0 + 780.0*1000.0, # semi-minor axis
    #                 "i": np.deg2rad(98.6), # inclination [rad]
    #                 }

    Target_Orbit["a"] = 0.5 * (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # semi-major axis
    Target_Orbit["e"] = (Target_Orbit["apogee"] - Target_Orbit["perigee"]) / (Target_Orbit["apogee"] + Target_Orbit["perigee"]) # eccentricity

    RecoveryStrategy = 'ASDS' # 'None', 'RTLS', 'ASDS'
    # define the optimization problem
    
    # opti = ca.Opti()
    optimization = LV_Opt.LV_Optimization(Target_Orbit=Target_Orbit, atmosphere=atmosphere)

    # opti = optimization.Build_Optimization(RecoveryStrategy = 'None', n_iter=750)
    opti = optimization.Build_Optimization(RecoveryStrategy = 'ASDS', n_iter=1000)

    # opti = optimization.Build_Optimization(RecoveryStrategy = 'RTLS', n_iter=0)

    WarmStart = optimization.Calc_WarmStart(RecoveryStrategy=RecoveryStrategy)

    opti = optimization.Set_WarmStart(opti, WarmStart)

    
    # Solve the optimization problem
    try:
        sol = opti.solve()

        payload_mass_sol = np.array(sol.value(optimization.sym_vars['payload_mass']))
        LaunchAz_sol = np.array(sol.value(optimization.sym_vars['LaunchAz']))
        u1_sol = np.array(sol.value(optimization.sym_vars['u1']))
        x1_sol = np.array(sol.value(optimization.sym_vars['x1']))
        dt1_sol = np.array(sol.value(optimization.sym_vars['dt1']))
        u2_sol = np.array(sol.value(optimization.sym_vars['u2']))
        x2_sol = np.array(sol.value(optimization.sym_vars['x2']))
        dt2_sol = np.array(sol.value(optimization.sym_vars['dt2']))
        dt3_sol = np.array(sol.value(optimization.sym_vars['dt3']))
        x3_sol = np.array(sol.value(optimization.sym_vars['x3']))
        u3_sol = np.array(sol.value(optimization.sym_vars['u3']))
        
        # propellent_mass_for_final_dv_sol = np.array(sol.value(propellent_mass_for_final_dv))
        if RecoveryStrategy != 'None':
            x4_0_sol = np.array(sol.value(optimization.sym_vars['x4_0']))
            dt4_ballistic_sol = np.array(sol.value(optimization.sym_vars['dt4_ballistic']))
            x4_before_reentry_sol = np.array(sol.value(optimization.sym_vars['x4_before_reentry']))
            x4_after_reentry_sol = np.array(sol.value(optimization.sym_vars['x4_after_reentry']))
            u4_reentry_sol = np.array(sol.value(optimization.sym_vars['u4_reentry']))
            dt4_reentry_sol = np.array(sol.value(optimization.sym_vars['dt4_reentry']))
            dt4_before_landing_sol = np.array(sol.value(optimization.sym_vars['dt4_before_landing']))
            x4_before_landing_sol = np.array(sol.value(optimization.sym_vars['x4_before_landing']))
            dt4_landing_sol = np.array(sol.value(optimization.sym_vars['dt4_landing']))
            x4_landing_sol = np.array(sol.value(optimization.sym_vars['x4_landing']))
            u4_landing_sol = np.array(sol.value(optimization.sym_vars['u4_landing']))
            if RecoveryStrategy == 'RTLS':
                dt4_boostback_sol = np.array(sol.value(optimization.sym_vars['dt4_boostback']))
                x4_boostback_sol = np.array(sol.value(optimization.sym_vars['x4_boostback']))
                u4_boostback_sol = np.array(sol.value(optimization.sym_vars['u4_boostback']))

    except:

        payload_mass_sol = opti.debug.value(opti.sym_vars['payload_mass'])
        LaunchAz_sol = opti.debug.value(opti.sym_vars['LaunchAz'])
        u1_sol = opti.debug.value(opti.sym_vars['u1'])
        x1_sol = opti.debug.value(opti.sym_vars['x1'])
        dt1_sol = opti.debug.value(opti.sym_vars['dt1'])
        u2_sol = opti.debug.value(opti.sym_vars['u2'])
        x2_sol = opti.debug.value(opti.sym_vars['x2'])
        dt2_sol = opti.debug.value(opti.sym_vars['dt2'])
        dt3_sol = opti.debug.value(opti.sym_vars['dt3'])
        x3_sol = opti.debug.value(opti.sym_vars['x3'])
        u3_sol = opti.debug.value(opti.sym_vars['u3'])
        # propellent_mass_for_final_dv_sol = np.array(opti.debug.value(propellent_mass_for_final_dv))
        if RecoveryStrategy != 'None':
            x4_0_sol = opti.debug.value(opti.sym_vars['x4_0'])
            dt4_ballistic_sol = opti.debug.value(opti.sym_vars['dt4_ballistic'])
            x4_before_reentry_sol = opti.debug.value(opti.sym_vars['x4_before_reentry'])
            x4_after_reentry_sol = opti.debug.value(opti.sym_vars['x4_after_reentry'])
            u4_reentry_sol = opti.debug.value(opti.sym_vars['u4_reentry'])
            dt4_reentry_sol = opti.debug.value(opti.sym_vars['dt4_reentry'])
            dt4_before_landing_sol = opti.debug.value(opti.sym_vars['dt4_before_landing'])
            x4_before_landing_sol = opti.debug.value(opti.sym_vars['x4_before_landing'])
            dt4_landing_sol = opti.debug.value(opti.sym_vars['dt4_landing'])
            x4_landing_sol = opti.debug.value(opti.sym_vars['x4_landing'])
            u4_landing_sol = opti.debug.value(opti.sym_vars['u4_landing'])
            if RecoveryStrategy == 'RTLS':
                dt4_boostback_sol = opti.debug.value(opti.sym_vars['dt4_boostback'])
                x4_boostback_sol = opti.debug.value(opti.sym_vars['x4_boostback'])
                u4_boostback_sol = opti.debug.value(opti.sym_vars['u4_boostback'])
    
    if 1:
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


    t1_vec = np.linspace(optimization.init_time, optimization.init_time+optimization.N1*dt1_sol, optimization.N1+1)
    t2_vec = np.linspace(t1_vec[-1], t1_vec[-1]+optimization.N2*dt2_sol, optimization.N2+1)
    t3_vec = np.linspace(t2_vec[-1], t2_vec[-1]+optimization.N3*dt3_sol, optimization.N3+1)

    Isp1_sol = np.zeros((optimization.N1))
    Qdyn1_sol = np.zeros((optimization.N1+1))
    a1_sol, b1_sol = np.zeros((optimization.N1+1)), np.zeros((optimization.N1+1))
    apogee1_sol, perigee1_sol = np.zeros((optimization.N1+1)), np.zeros((optimization.N1+1))
    alt1_sol, vel1_sol = np.zeros((optimization.N1+1)), np.zeros((optimization.N1+1))
    i1_sol, gama1_sol, gama1_local_sol = np.zeros((optimization.N1+1)), np.zeros((optimization.N1+1)), np.zeros((optimization.N1+1))
    x1_eci_sol = np.zeros((rocket_eci.nx, optimization.N1+1))
    for k in range(optimization.N1+1):
        if k < optimization.N1:
            Isp1_sol[k] = rocket_Booster.ISP_calc(x1_sol[:,k], u1_sol[:,k]).full().flatten()[0]
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

    Isp2_sol = np.zeros((optimization.N2))
    Alpha2_sol = np.zeros((optimization.N2))
    a2_sol, b2_sol = np.zeros((optimization.N2+1)), np.zeros((optimization.N2+1))
    apogee2_sol, perigee2_sol = np.zeros((optimization.N2+1)), np.zeros((optimization.N2+1))
    alt2_sol, i2_sol = np.zeros((optimization.N2+1)), np.zeros((optimization.N2+1))
    Qdyn2_sol, gama2_sol = np.zeros((optimization.N2+1)), np.zeros((optimization.N2+1))
    for k in range(optimization.N2+1):
        if k < optimization.N2:
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

    Isp3_sol = np.zeros((optimization.N3))
    Alpha3_sol = np.zeros((optimization.N3))
    a3_sol, b3_sol = np.zeros((optimization.N3+1)), np.zeros((optimization.N3+1))
    apogee3_sol, perigee3_sol = np.zeros((optimization.N3+1)), np.zeros((optimization.N3+1))
    alt3_sol, i3_sol, gama3_sol = np.zeros((optimization.N3+1)), np.zeros((optimization.N3+1)), np.zeros((optimization.N3+1))
    for k in range(optimization.N3+1):
        if k < optimization.N3:
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
    propellent_mass_for_final_dv3 = x3_sol[6,optimization.N3] * (1.0 - ca.exp((v3_apogee - v3_desired) / (rocket_eci.SecondStage_Vac_Isp * rocket_eci.g0)))
    actual_apogee3 = Target_Orbit["a"] * (1.0 + Target_Orbit["e"])

    if RecoveryStrategy != 'None':
        alt4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[0], rocket_return.R0+x4_0_sol[1])) - rocket_return.R0
        vel4_0_sol = ca.norm_2(ca.vertcat(x4_0_sol[2], x4_0_sol[3]))
        
        if RecoveryStrategy == 'RTLS':
            t4_boostback_vec = t1_vec[-1]+np.linspace(0, dt4_boostback_sol, optimization.N4+1)
            alt4_boostback_sol = np.zeros((optimization.N4+1,1))
            vel4_boostback_sol = np.zeros((optimization.N4+1,1))
            x4_boostback_vec_sol = np.zeros((rocket_boostback.nx, optimization.N4+1))
            x4_boostback_vec_sol[:,0] = x4_0_sol
            for k in range(optimization.N4+1):
                if k<optimization.N4:
                    x4_boostback_vec_sol[:,k+1] = rocket_boostback.dynamics_kp1(x4_boostback_vec_sol[:,k], u4_boostback_sol, dt4_boostback_sol/optimization.N4).full().flatten()
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
        t4_ballistic_vec[0]  = t1_vec[-1] if RecoveryStrategy != 'RTLS' else dt4_boostback_sol+t1_vec[-1]
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
            t4_before_landing_vec[k] = t4_before_landing_vec[k-1] + dt4_before_landing_sol/(N4_before_landing-1)
            x4_before_landing_vec_sol[:,k] = rocket_return.dynamics_kp1(x4_before_landing_vec_sol[:,k-1], 0, dt4_before_landing_sol/(N4_before_landing-1)).full().flatten()
            alt4_before_landing_sol[k] = rocket_return.local_to_alt(x4_before_landing_vec_sol[:,k])
            vel4_before_landing_sol[k] = ca.norm_2(ca.vertcat(x4_before_landing_vec_sol[2,k], x4_before_landing_vec_sol[3,k]))
            Qdyn4_before_landing_sol[k] = 0.5 * atmosphere.rho_fun(alt4_before_landing_sol[k]) * vel4_before_landing_sol[k]**2
        
        # landing burn phase
        N4_landing = optimization.N4
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
    if RecoveryStrategy != 'RTLS':
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
    ax_ap.set_yscale('symlog', linthresh=1)


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
    if RecoveryStrategy == 'RTLS':
        ax_mass.plot(t4_boostback_vec, x4_boostback_vec_sol[4,:]/1000, label='Boostback')
        ax_mass.plot(dt4_boostback_sol, x4_boostback_sol[4]/1000,'s', label='Booster Return')
    if RecoveryStrategy != 'None':
        ax_mass.plot(t4_ballistic_vec, x4_ballistic_sol[4,:]/1000)
        ax_mass.plot(t4_reentry_vec, x4_reentry_sol[4,:]/1000)
        ax_mass.plot(t4_before_landing_vec, x4_before_landing_vec_sol[4,:]/1000)
        ax_mass.plot(t4_landing_vec, x4_landing_sol[4,:]/1000)
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.FirstStage_EmptyMass)/1000, (payload_mass_sol+rocket_Booster.FirstStage_EmptyMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.LV_total_mass)/1000, (payload_mass_sol+rocket_Booster.LV_total_mass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.SecondStage_FullMass)/1000, (payload_mass_sol+rocket_Booster.SecondStage_FullMass)/1000], 'k--')
    ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+rocket_Booster.SecondStage_EmptyMass)/1000, (payload_mass_sol+rocket_Booster.SecondStage_EmptyMass)/1000], 'k--')
    # ax_mass.plot(t3_vec[-1], (x3_sol[6,N3]-propellent_mass_for_final_dv_sol)/1000, 's', label='Mass After Final dV')
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
    ax_gama.plot(t1_vec, np.rad2deg(gama1_local_sol), label='1st Stage Local')
    # ax_gama.plot(t1_vec, np.rad2deg(np.atan2(x1_guess[3,:], x1_guess[2,:])), label='1st Stage Local guess')
    ax_gama.plot(t2_vec, np.rad2deg(gama2_sol), label='2nd Stage before fairing separation')
    ax_gama.plot(t3_vec, np.rad2deg(gama3_sol), label='2nd Stage after fairing separation') 
    ax_gama.set_title('Gama')
    ax_gama.set_ylabel('Gama [deg]')
    ax_gama.set_xlabel('Time [s]')
    ax_gama.grid()
    ax_gama.legend(loc='best')

    plt.tight_layout()

    # Calc parking orbit
    x_parking = x3_sol[:,-1].reshape(1,-1)
    dtheta = 0
    tf = 0
    while True:
        x_parking = np.vstack((x_parking, rocket_eci.dynamics_kp1(x_parking[-1,:], 0, 1.0).full().flatten()))
        tf += 1.0
        dtheta += np.arccos(np.dot(x_parking[-1,0:3],x_parking[-2,0:3])/(np.linalg.norm(x_parking[-1,0:3])*np.linalg.norm(x_parking[-2,0:3])))
        if dtheta >= 2*np.pi:
            break

    r_parking = np.linalg.norm(x_parking[:,0:3], axis=1)
    i_apogee = np.argmax(r_parking)
    x_final = x_parking[i_apogee,:].reshape(1,-1)
    x_final[-1,3:6] = v3_desired * x_final[-1,3:6] / np.linalg.norm(x_final[-1,3:6])

    Parking_apogee = np.linalg.norm(x_parking[np.argmax(r_parking),0:3])
    Parking_perigee = np.linalg.norm(x_parking[np.argmin(r_parking),0:3])

    
    dtheta = 0
    while True:
        x_final = np.vstack((x_final, rocket_eci.dynamics_kp1(x_final[-1,:], 0, dt3_sol).full().flatten()))
        dtheta += np.arccos(np.dot(x_final[-1,0:3],x_final[-2,0:3])/(np.linalg.norm(x_final[-1,0:3])*np.linalg.norm(x_final[-2,0:3])))
        if dtheta >= 2*np.pi:
            break
    r_final = np.linalg.norm(x_final[:,0:3], axis=1)

    final_apogee = np.linalg.norm(x_final[np.argmax(r_final),0:3])
    final_perigee = np.linalg.norm(x_final[np.argmin(r_final),0:3])
        


    phi, theta = np.mgrid[0:np.pi:30j, 0:2*np.pi:30j]
    x = rocket_Booster.R0 * np.sin(phi) * np.cos(theta)
    y = rocket_Booster.R0 * np.sin(phi) * np.sin(theta)
    z = rocket_Booster.R0 * np.cos(phi)

    # Create equatorial plane (z=0)
    theta_plane = np.linspace(0, 2*np.pi, 100)
    r_plane = np.linspace(0, rocket_Booster.R0*1.1, 25)
    r_grid, theta_grid = np.meshgrid(r_plane, theta_plane)
    x_plane = r_grid * np.cos(theta_grid)
    y_plane = r_grid * np.sin(theta_grid)
    z_plane = np.zeros_like(x_plane)
    
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


    plt.show()
        




    
