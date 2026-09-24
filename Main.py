#!/usr/bin/env pythorocket_eci.N[1]

import os
os.system('clear')
from dataclasses import fields

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ProcessPoolExecutor
import matplotlib.gridspec as gridspec

import LV_Optimization_Type as LVopt_Type
from LV_Type_scaled import DispesrionFactorsType

# Global figures/axes used by the plotting helpers
fig_first_stage = None
fig_second_stage = None
fig_return_stage = None
ax1_traj = ax1_vel = ax1_input = ax1_Isp = ax1_mass = ax1_q = None
ax2_vel = ax2_input = ax2_mass = ax2_alt = ax2_inc = ax2_ap = None
ax3_traj = ax3_vel = ax3_alt = ax3_heat = ax3_mass = ax3_q = None


def ensure_first_stage_axes():
    """Lazily create the first-stage plotting axes."""
    global fig_first_stage, ax1_traj, ax1_vel, ax1_input, ax1_Isp, ax1_mass, ax1_q
    if ax1_traj is not None:
        return
    fig_first_stage, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax1_traj, ax1_vel, ax1_input = axes[0]
    ax1_Isp, ax1_mass, ax1_q = axes[1]


def ensure_second_stage_axes():
    """Lazily create the second-stage plotting axes."""
    global fig_second_stage, ax2_vel, ax2_input, ax2_mass, ax2_alt, ax2_inc, ax2_ap
    if ax2_vel is not None:
        return
    fig_second_stage, axes = plt.subplots(3, 2, figsize=(14, 12))
    (ax2_vel, ax2_input), (ax2_mass, ax2_alt), (ax2_inc, ax2_ap) = axes


def ensure_return_stage_axes():
    """Lazily create the return-stage plotting axes."""
    global fig_return_stage, ax3_traj, ax3_vel, ax3_alt, ax3_heat, ax3_mass, ax3_q
    if ax3_traj is not None:
        return
    fig_return_stage, axes = plt.subplots(3, 2, figsize=(14, 12))
    (ax3_traj, ax3_vel), (ax3_alt, ax3_heat), (ax3_mass, ax3_q) = axes


def plot_First_Stage():
    ensure_first_stage_axes()

    ax1_traj.plot(Solution['x1'][0,:]/1000, Solution['x1'][1,:]/1000, color=col, label='1st Stage', linewidth=2)
    ax1_traj.set_title('Trajectory', fontsize=20)
    ax1_traj.set_ylabel('Altitude [Km]', fontsize=20)
    ax1_traj.set_xlabel('Distance [Km]', fontsize=20)
    ax1_traj.set_aspect('equal', adjustable='box')
    ax1_traj.grid('on')


    ax1_vel.plot(Solution['t1_vec'], np.linalg.norm(Solution['x1'][2:4,:], axis=0), color=col, label='1st Stage Local', linewidth=2)
    ax1_vel.set_title('Velocity Local Frame', fontsize=20)
    ax1_vel.set_ylabel('Velocity [m/s]', fontsize=20)
    ax1_vel.set_xlabel('Time [s]', fontsize=20)
    ax1_vel.grid('on')

    if RecoveryStrategy == 'EXP':
        ax1_input.plot(Solution['t1_vec'][:-1], Solution['u1'][0,:], color=col, linewidth=2, label=r'$f_T$')
        ax1_input.plot(Solution['t1_vec'][:-1], np.rad2deg(Solution['u1'][1,:])/15, '--', color=col, linewidth=2, label=r'$\alpha / \alpha_{\max}$')
    else:
        ax1_input.plot(Solution['t1_vec'][:-1], Solution['u1'][0,:], color=col, linewidth=2)
        ax1_input.plot(Solution['t1_vec'][:-1], np.rad2deg(Solution['u1'][1,:])/15, '--', color=col, linewidth=2)        
    ax1_input.set_title(r'Inputs: Thrust Factor and $\alpha / \alpha_{\max}$', fontsize=20)
    ax1_input.set_ylabel(r'$f_T$ and $\alpha / \alpha_{\max}$ [-]', fontsize=20)
    ax1_input.set_xlabel('Time [s]', fontsize=20)
    ax1_input.grid('on')
    ax1_input.set_ylim(-0.5, 1.1)
    ax1_input.legend(fontsize = 15)

    ax1_Isp.plot(Solution['t1_vec'][:-1], Solution['Isp1_sol'], color=col, linewidth=2)
    ax1_Isp.set_title('Specific Thrust [Sec]', fontsize=20)
    ax1_Isp.set_ylabel('Isp [sec]', fontsize=20)
    ax1_Isp.set_xlabel('Time [s]', fontsize=20)
    ax1_Isp.grid('on')

    mass = Solution['payload_mass']
    ax1_mass.plot(Solution['t1_vec'], Solution['x1'][4,:]/1000, color=col, linewidth=2, label = f'{RecoveryStrategy}, Payload = {mass/1000:.4}[Tons]')
    ax1_mass.set_title('Mass', fontsize=20)
    ax1_mass.set_ylabel('Mass [Tons]', fontsize=20)
    ax1_mass.set_xlabel('Time [s]', fontsize=20)
    ax1_mass.grid('on')
    ax1_mass.legend(fontsize = 15)


    ax1_q.plot(Solution['t1_vec'], Solution['Qdyn1_sol']/1000.0, color=col, linewidth=2, label='1st Stage')
    ax1_q.set_title('Qynamic Pressure', fontsize=20)
    ax1_q.set_ylabel('Qynamic Pressure [KPa]', fontsize=20)
    ax1_q.set_xlabel('Time [s]', fontsize=20)
    ax1_q.grid('on')

    plt.savefig("FirstStage_Dynamics.png", dpi=300, bbox_inches="tight")

def plot_Second_Stage():
    ensure_second_stage_axes()

    ax2_vel.plot(Solution['t2_vec'], np.linalg.norm(Solution['x2'][3:6,:], axis=0), color=col, linewidth=2)
    ax2_vel.plot(Solution['t3_vec'], np.linalg.norm(Solution['x3'][3:6,:], axis=0), color=col, linewidth=2)
    ax2_vel.set_title('Velocity ECI Frame', fontsize=20)
    ax2_vel.set_ylabel('Velocity [m/s]', fontsize=20)
    ax2_vel.set_xlabel('Time [s]', fontsize=20)
    ax2_vel.grid('on')

    if RecoveryStrategy == 'EXP':
        ax2_input.plot(Solution['t2_vec'][:-1], np.linalg.norm(Solution['u2'], axis=0)/LVopt.eci.SecondStage_Thrust, color=col, linewidth=2, label=r'$f_T$')
        ax2_input.plot(Solution['t2_vec'][:-1], np.rad2deg(Solution['Alpha2_sol'])/20,'--', color=col, linewidth=2, label=r'$\alpha / 20^0$')    
    else:
        ax2_input.plot(Solution['t2_vec'][:-1], np.linalg.norm(Solution['u2'], axis=0)/LVopt.eci.SecondStage_Thrust, color=col, linewidth=2)
        ax2_input.plot(Solution['t2_vec'][:-1], np.rad2deg(Solution['Alpha2_sol'])/20,'--', color=col, linewidth=2)    
    ax2_input.plot(Solution['t3_vec'][:-1], np.linalg.norm(Solution['u3'], axis=0)/LVopt.eci.SecondStage_Thrust, color=col, linewidth=2)
    ax2_input.plot(Solution['t3_vec'][:-1], np.rad2deg(Solution['Alpha3_sol'])/20,'--', color=col, linewidth=2)
    # ax2_thr.set_ylim(-0.0, 1.1)
    
    ax2_input.set_title(r'Input: Thrust and $\alpha$', fontsize=20)
    ax2_input.set_ylabel(r'$f_T$ and $\alpha$ [deg]', fontsize=20)
    ax2_input.set_xlabel('Time [s]', fontsize=20)
    ax2_input.grid('on')
    ax2_input.legend(fontsize = 15)

    mass = Solution['payload_mass']
    ax2_mass.plot(Solution['t2_vec'], Solution['x2'][6,:]/1000, color=col, linewidth=2, label = f'{RecoveryStrategy}, Payload = {mass/1000:.4}[Tons]')
    ax2_mass.plot(Solution['t3_vec'], Solution['x3'][6,:]/1000, color=col, linewidth=2)
    ax2_mass.set_title('Mass', fontsize=20)
    ax2_mass.set_ylabel('Mass [Tons]', fontsize=20)
    ax2_mass.set_xlabel('Time [s]', fontsize=20)
    ax2_mass.grid('on')
    ax2_mass.legend(fontsize = 15)

    ax2_alt.plot(Solution['t2_vec'], Solution['alt2_sol']/1000, color=col)
    ax2_alt.plot(Solution['t3_vec'], Solution['alt3_sol']/1000, color=col)
    ax2_alt.set_title('Altitude', fontsize=20)
    ax2_alt.set_ylabel('Altitude [Km]', fontsize=20)
    ax2_alt.set_xlabel('Time [s]', fontsize=20)
    ax2_alt.grid('on')

    ax2_inc.plot(Solution['t2_vec'], np.rad2deg(Solution['i2_sol']), color=col, label='2nd Stage after fairing separation')
    ax2_inc.plot(Solution['t3_vec'], np.rad2deg(Solution['i3_sol']), color=col, label='2nd Stage after fairing separation')
    ax2_inc.plot([0, Solution['t3_vec'][-1]], [np.rad2deg(Target_Orbit['i']), np.rad2deg(Target_Orbit['i'])], 'k--')
    ax2_inc.set_title('Inclination')
    ax2_inc.set_ylabel('Inclination [deg]')
    ax2_inc.set_xlabel('Time [s]')
    ax2_inc.grid('on')

    if RecoveryStrategy == 'EXP':
        ax2_ap.plot(Solution['t3_vec'], (Solution['perigee3_sol'] - R0)/1000,'--', color=col, label='Perigee')
        ax2_ap.plot(Solution['t3_vec'], (Solution['apogee3_sol'] - R0)/1000, color=col, label='Apogee')
        ax2_ap.plot([0, Solution['t3_vec'][-1]], [(Target_Orbit['apogee'] - R0)/1000, (Target_Orbit['apogee'] - R0)/1000], 'k--', label='Target Apogee')
        ax2_ap.plot([0, Solution['t3_vec'][-1]], [(Target_Orbit['perigee'] - R0)/1000, (Target_Orbit['perigee'] - R0)/1000], 'c--', label='Target Perigee')
        ax2_ap.plot([0, Solution['t3_vec'][-1]], [LVopt.eci.ParkingOrbit_PerigeeAlt/1000, LVopt.eci.ParkingOrbit_PerigeeAlt/1000], 'm--', label='Parking Perigee')
    else:
        ax2_ap.plot(Solution['t3_vec'], (Solution['perigee3_sol'] - R0)/1000,'--', color=col)
        ax2_ap.plot(Solution['t3_vec'], (Solution['apogee3_sol'] - R0)/1000, color=col)
    
    ax2_ap.plot(Solution['t2_vec'], (Solution['perigee2_sol'] - R0)/1000,'--', color=col)
    ax2_ap.plot(Solution['t2_vec'], (Solution['apogee2_sol'] - R0)/1000, color=col)
    ax2_ap.set_title('Apogee/Perigee Altitude')
    ax2_ap.set_ylabel('Apogee/Perigee Altitude [Km]')
    ax2_ap.set_xlabel('Time [s]')
    ax2_ap.grid('on')
    ax2_ap.set_yscale('symlog', linthresh=100)
    ax2_ap.legend(fontsize = 15, loc='lower left')

    plt.savefig("SecondStage_Dynamics.png", dpi=300, bbox_inches="tight")


def plot_Return_Stage():
    ensure_return_stage_axes()

    if RecoveryStrategy == 'EXP': return

    ax3_traj.plot(Solution['x1'][0,:]/1000, Solution['x1'][1,:]/1000, color=col, linewidth=2)
    if RecoveryStrategy == 'ASDS':
        x_earth = np.linspace(0, np.max(Solution['x4'][:,0])*1.1,1000)
        z_earth = np.sqrt(R0**2 - x_earth**2) - R0
        ax3_traj.plot(x_earth/1000, z_earth/1000, 'k--')

    ax3_traj.plot(Solution['x4'][:,0]/1000, Solution['x4'][:,1]/1000, color=col)
    ax3_traj.set_title('Trajectory', fontsize=20)
    ax3_traj.set_ylabel('Z Local [Km]', fontsize=20)
    ax3_traj.set_xlabel('X Local [Km]', fontsize=20)
    ax3_traj.set_aspect('equal', adjustable='box')
    ax3_traj.grid('on')


    ax3_vel.plot(Solution['t1_vec'], np.linalg.norm(Solution['x1'][2:4,:], axis=0), color=col, linewidth=2)
    ax3_vel.plot(Solution['t4_vec'], np.linalg.norm(Solution['x4'][:,2:4], axis=1), color=col, linewidth=2)
    ax3_vel.set_title('Velocity Local Frame', fontsize=20)
    ax3_vel.set_ylabel('Velocity [m/s]', fontsize=20)
    ax3_vel.set_xlabel('Time [s]', fontsize=20)
    ax3_vel.grid('on')

    ax3_alt.plot(Solution['t1_vec'], Solution['alt1_sol']/1000, color=col, linewidth=2)
    ax3_alt.plot(Solution['t4_vec'], Solution['alt4_sol']/1000, color=col, linewidth=2)
    ax3_alt.set_title('Altitude', fontsize=20)
    ax3_alt.set_ylabel('Altitude [Km]', fontsize=20)
    ax3_alt.set_xlabel('Time [s]', fontsize=20)
    ax3_alt.grid('on')


    Alt = np.sqrt(Solution['x4'][:,0]**2 + (R0 + Solution['x4'][:,1])**2) - R0
    rho = LVopt.atmosphere.rho_fun(Alt).full().flatten()
    ax3_heat.plot(Solution['t4_vec'], 0.001*LVopt.booster.Booster_k_empirical * np.sqrt(rho) * np.linalg.norm(Solution['x4'][:,2:4], axis=1) ** 3, color=col, linewidth=2)
    ax3_heat.set_title('Empirical Heat Flux', fontsize=20)
    ax3_heat.set_ylabel('Heat Flux [KWatt/m^2]', fontsize=20)
    ax3_heat.set_xlabel('Time [s]', fontsize=20)
    ax3_heat.grid('on')

    mass = Solution['payload_mass']
    ax3_mass.plot(Solution['t4_vec'], Solution['x4'][:,4]/1000, color=col, linewidth=2, label = f'{RecoveryStrategy}, Payload = {mass/1000:.4}[Tons]')
    ax3_mass.set_title('Mass', fontsize=20)
    ax3_mass.set_ylabel('Mass [Tons]', fontsize=20)
    ax3_mass.set_xlabel('Time [s]', fontsize=20)
    ax3_mass.grid('on')
    ax3_mass.legend(fontsize = 15)


    ax3_q.plot(Solution['t4_vec'], Solution['Qdyn4_sol']/1000.0, color=col, linewidth=2, label='1st Stage')
    ax3_q.set_title('Qynamic Pressure', fontsize=20)
    ax3_q.set_ylabel('Qynamic Pressure [KPa]', fontsize=20)
    ax3_q.set_xlabel('Time [s]', fontsize=20)
    ax3_q.grid('on')

    plt.savefig("ReturnStage_Dynamics.png", dpi=300, bbox_inches="tight")



R0: float = 6378137.0 # [m]
color_vec = ['r','b','g','k','y','c','m']


def run_dispersion_case(i):
    """Run one sweep and return summary lines for printing after all sweeps finish."""
    global LVopt, Solution, col, icol, RecoveryStrategy, Target_Orbit
    DispesrionVec = DispesrionList[i]
    DispesrionParam = DispesrionParamList[i]
    LV_Configuration = ConfigList[i]
    # Keep the historical sweep name as an alias for the dataclass field.
    dispersion_field = {'StagePartition': 'StagePartitionDelta'}.get(DispesrionParam, DispesrionParam)
    if dispersion_field not in {item.name for item in fields(DispesrionFactorsType)}:
        raise ValueError(f'Unknown dispersion parameter: {DispesrionParam}')

    result_lines = []
    for iOrbit, Target_Orbit in enumerate(Target_Orbits):
        print('#'+'='*40+f" Orbit Name: {Target_Orbit['Name']} "+"="*40)
        for iRecover, RecoveryStrategy in enumerate(RecoveryStrategyVec):
            Results = []
            init_guess = 0.5
            for iVal, DispesrionValue in enumerate(DispesrionVec):
                DispesrionFactors = DispesrionFactorsType()
                setattr(DispesrionFactors, dispersion_field, float(DispesrionValue))
                LVopt = LVopt_Type.LV_Optimization(DispesrionFactors, LV_Configuration)
                Solution = {'success': False, 'return_status': 'Not attempted'}

                for _ in range(5):
                    try:
                        Solution = LVopt.SolveOptimiztion(RecoveryStrategy=RecoveryStrategy,
                                                            payload_mass_predefined=payload_mass_predefined,
                                                            Target_Orbit=Target_Orbit,
                                                            Plot_interm=Plot_interm,
                                                            init_guess=init_guess)
                        if Solution['success']:
                            init_guess = Solution
                            break
                        else:
                            init_guess = np.random.uniform(0.2, 0.8)
                    except Exception as exc:
                        Solution = {'success': False, 'return_status': str(exc)}
                        print(f'{RecoveryStrategy} attempt failed: {exc}')
                        init_guess = np.random.uniform(0.2, 0.8)

                else:
                    print(f"{RecoveryStrategy} failed: {Solution.get('return_status', 'Unknown solver failure')}")

                if Solution['success']:
                    Results.append(Solution['payload_mass'])
                else:
                    Results.append(0.0)
                if iVal == DispesrionVec.shape[0]-1:
                    ConfigName = 'Falcon9' if LV_Configuration==1 else 'Starship'
                    result_lines.append(ConfigName+'_'+Target_Orbit['Name']+'_'+RecoveryStrategy+'_'+DispesrionParam+'=0.001*np.array(['+', '.join(f'{x:.2f}' for x in Results) + '])')
                # print(f'dV for final orbit: {v3_desired-v3_apogee:.2f} m/s, Propellent mass for final dV: {propellent_mass_for_final_dv3:.2f} kg')
                # print(f'Final Orbit (Apogee, Perigee, Inclination): {(actual_apogee3 - self.eci.R0)/1000:.2f} Km, {(apogee3_sol[-1] - self.eci.R0)/1000:.2f} Km, {np.rad2deg(i3_sol[-1]):.2f} deg')
                # print(f'Parking Orbit (Apogee, Perigee, Inclination): {(apogee3_sol[-1] - self.eci.R0)/1000:.2f} Km, {(perigee3_sol[-1] - self.eci.R0)/1000:.2f} Km, {np.rad2deg(i3_sol[-1]):.2f} deg')
                # print('='*80)

                if Plot_Results == 0 or not Solution['success']: continue
                col = color_vec[icol]
                icol = (1+icol)%len(color_vec)
                ax_traj.plot(Solution['x1'][0,:]/1000, Solution['x1'][1,:]/1000, color=col, label='1st Stage')
                if RecoveryStrategy != 'EXP':
                    if RecoveryStrategy == 'ASDS':
                        x_earth = np.linspace(0, np.max(Solution['x4'][:,0])*1.1,1000)
                        z_earth = np.sqrt(R0**2 - x_earth**2) - R0
                        ax_traj.plot(x_earth/1000, z_earth/1000, 'k--')

                    ax_traj.plot(Solution['x4'][:,0]/1000, Solution['x4'][:,1]/1000, color=col, label='Booster Return')
                ax_traj.set_title('Trajectory')
                ax_traj.set_ylabel('Altitude [Km]')
                ax_traj.set_xlabel('Distance [Km]')
                ax_traj.set_aspect('equal', adjustable='box')
                ax_traj.grid('on')

                ax_alt.plot(Solution['t1_vec'], Solution['alt1_sol']/1000, color=col, label='1st Stage')
                ax_alt.plot(Solution['t2_vec'], Solution['alt2_sol']/1000, color=col, label='2nd Stage before fairing separation')
                ax_alt.plot(Solution['t3_vec'], Solution['alt3_sol']/1000, color=col, label='2nd Stage after fairing separation')
                if RecoveryStrategy != 'EXP':
                    ax_alt.plot(Solution['t4_vec'], Solution['alt4_sol']/1000, color=col, label='Booster Return')

                ax_alt.set_title('Altitude')
                ax_alt.set_ylabel('Altitude [Km]')
                ax_alt.set_xlabel('Time [s]')
                ax_alt.grid('on')

                ax_incl.plot(Solution['t3_vec'], np.rad2deg(Solution['i3_sol']), color=col, label='2nd Stage after fairing separation')
                ax_incl.plot([0, Solution['t3_vec'][-1]], [np.rad2deg(Target_Orbit['i']+0.001), np.rad2deg(Target_Orbit['i']+0.001)], 'k--')
                ax_incl.plot([0, Solution['t3_vec'][-1]], [np.rad2deg(Target_Orbit['i']-0.001), np.rad2deg(Target_Orbit['i']-0.001)], 'k--')
                ax_incl.plot([0, Solution['t3_vec'][-1]], [np.rad2deg(Target_Orbit['i']), np.rad2deg(Target_Orbit['i'])], 'k--')
                ax_incl.set_title('Inclination')
                ax_incl.set_ylabel('Inclination [deg]')
                ax_incl.set_xlabel('Time [s]')
                ax_incl.grid('on')

                ax_ap.plot(Solution['t3_vec'], (Solution['perigee3_sol'] - R0)/1000, color=col, label='Perigee')
                ax_ap.plot(Solution['t3_vec'], (Solution['apogee3_sol'] - R0)/1000, color=col, label='Apogee')
                ax_ap.plot([0, Solution['t3_vec'][-1]], [(Target_Orbit['apogee'] - R0)/1000, (Target_Orbit['apogee'] - R0)/1000], 'k--')
                ax_ap.plot([0, Solution['t3_vec'][-1]], [(Target_Orbit['perigee'] - R0)/1000, (Target_Orbit['perigee'] - R0)/1000], 'k--')
                ax_ap.set_title('Apogee/Perigee Altitude')
                ax_ap.set_ylabel('Apogee/Perigee [Km]')
                ax_ap.set_xlabel('Time [s]')
                ax_ap.grid('on')
                ax_ap.set_yscale('symlog', linthresh=100)

                ax_vel.plot(Solution['t1_vec'], np.linalg.norm(Solution['x1'][2:4,:], axis=0), color=col, label='1st Stage Local')
                ax_vel.plot(Solution['t2_vec'], np.linalg.norm(Solution['x2'][3:6,:], axis=0), color=col, label='2nd Stage ECI')
                ax_vel.plot(Solution['t3_vec'], np.linalg.norm(Solution['x3'][3:6,:], axis=0), color=col, label='2nd Stage ECI')
                if RecoveryStrategy != 'EXP':
                    ax_vel.plot(Solution['t4_vec'], np.linalg.norm(Solution['x4'][:,2:4], axis=1), color=col, label = 'Booster Local')
                ax_vel.set_title('Velocity')
                ax_vel.set_ylabel('Velocity [m/s]')
                ax_vel.set_xlabel('Time [s]')
                ax_vel.grid('on')

                ax_throttle.plot(Solution['t1_vec'][:-1], Solution['u1'][0,:], color=col)
                ax_throttle.plot(Solution['t2_vec'][:-1], np.linalg.norm(Solution['u2'], axis=0), color=col)
                ax_throttle.plot(Solution['t3_vec'][:-1], np.linalg.norm(Solution['u3'], axis=0), color=col)
                # if RecoveryStrategy != 'EXP':
                #     ax_throttle.plot(t4_ballistic_vec, 0*t4_ballistic_vec, label='Ballistic Phase')
                #     ax_throttle.plot(t4_reentry_vec, np.ones_like(t4_reentry_vec)*u4_reentry_scaled_sol, label='Reentry Phase')
                #     ax_throttle.plot(t4_landing_vec[:-1], u4_landing_scaled_sol.T, label='Landing Phase')
                ax_throttle.set_title('Thrust Factor')
                ax_throttle.set_ylabel('Thrust Factor [-]')
                ax_throttle.set_xlabel('Time [s]')
                ax_throttle.grid('on')
                ax_throttle.set_ylim(0, 1.1)

                ax_alpha.plot(Solution['t1_vec'][:-1], np.rad2deg(Solution['u1'][1,:]), color=col)
                ax_alpha.plot(Solution['t2_vec'][:-1], np.rad2deg(Solution['Alpha2_sol']), color=col)
                ax_alpha.plot(Solution['t3_vec'][:-1], np.rad2deg(Solution['Alpha3_sol']), color=col)
                ax_alpha.set_title('Angle of Attack')
                ax_alpha.set_ylabel('Angle of Attack [deg]')
                ax_alpha.set_xlabel('Time [s]')
                ax_alpha.grid('on')

                ax_Isp.plot(Solution['t1_vec'][:-1], Solution['Isp1_sol'], color=col)
                ax_Isp.plot(Solution['t2_vec'][:-1], Solution['Isp2_sol'], color=col)
                ax_Isp.plot(Solution['t3_vec'][:-1], Solution['Isp3_sol'], color=col)
                ax_Isp.set_title('Isp')
                ax_Isp.set_ylabel('Isp [sec]')
                ax_Isp.set_xlabel('Time [s]')
                ax_Isp.grid('on')

                mass = Solution['payload_mass']
                ax_mass.plot(Solution['t1_vec'], Solution['x1'][4,:]/1000, color=col, label = f'Payload = {mass/1000:.4}[Tons]')
                ax_mass.plot(Solution['t2_vec'], Solution['x2'][6,:]/1000, color=col)
                ax_mass.plot(Solution['t3_vec'], Solution['x3'][6,:]/1000, color=col)
                if RecoveryStrategy != 'EXP':
                    ax_mass.plot(Solution['t4_vec'], Solution['x4'][:,4]/1000, color=col)
                # ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.FirstStage_EmptyMass)/1000, (payload_mass_sol+booster.FirstStage_EmptyMass)/1000], 'k--')
                # ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.LV_total_mass)/1000, (payload_mass_sol+booster.LV_total_mass)/1000], 'k--')
                # ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.SecondStage_FullMass)/1000, (payload_mass_sol+booster.SecondStage_FullMass)/1000], 'k--')
                # ax_mass.plot([0, t3_vec[-1]], [(payload_mass_sol+booster.SecondStage_EmptyMass)/1000, (payload_mass_sol+booster.SecondStage_EmptyMass)/1000], 'k--')
                # ax_mass.plot(t3_vec[-1], (x3_sol[6,rocket_eci.N[1]]-propellent_mass_for_final_dv_sol)/1000, 's', label='Mass After Final dV')
                # if RecoveryStrategy != 'EXP':
                #     ax_mass.plot([0, t4_landing_vec[-1,0]], [(booster.EmptyFirstStageMass)/1000, (booster.EmptyFirstStageMass)/1000], 'k--')
                ax_mass.set_title('Mass')
                ax_mass.set_ylabel('Mass [Tons]')
                ax_mass.set_xlabel('Time [s]')
                ax_mass.grid('on')
                ax_mass.legend()

                ax_acc.plot(Solution['t1_vec'][:-1], Solution['acc1_sol'], color=col)
                ax_acc.plot(Solution['t2_vec'][:-1], Solution['acc2_sol'], color=col)
                ax_acc.plot(Solution['t3_vec'][:-1], Solution['acc3_sol'], color=col)
                if RecoveryStrategy != 'EXP':
                    ax_acc.plot(Solution['t4_vec'], Solution['acc4_sol'],'-.', color=col)
                ax_acc.set_title('Specific Acceleration')
                ax_acc.set_ylabel('Specific Acceleration [m/s^2]')
                ax_acc.set_xlabel('Time [s]')
                ax_acc.grid('on')

                ax_q.plot(Solution['t1_vec'], Solution['Qdyn1_sol']/1000.0, color=col, label='1st Stage')
                if RecoveryStrategy != 'EXP':
                    ax_q.plot(Solution['t4_vec'], Solution['Qdyn4_sol']/1000.0, color=col, label='Booster Return')

                    # x4_before_landing_intren_sol = x4_after_reentry_scaled_sol
                    # for i in range(N_before_landing_constraints):
                    #     x4_before_landing_intren_sol = rocket_return.dynamics_kp1(x4_before_landing_intren_sol, 0, dt4_before_landing_sol/N_before_landing_constraints/rocket_return.scaleT)
                    #     x4_before_landing_intren_sol_unscaled = rocket_return.unscale_x(x4_before_landing_intren_sol)
                    #     Qdyn = 0.5 * atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren_sol_unscaled)) * (x4_before_landing_intren_sol_unscaled[2]**2 + x4_before_landing_intren_sol_unscaled[3]**2)
                    #     ax_q.plot(t4_reentry_vec[-1] + (i+1)*dt4_before_landing_sol/N_before_landing_constraints, Qdyn/1000.0, 'ro')

                ax_q.set_title('Qynamic Pressure')
                ax_q.set_ylabel('Qynamic Pressure [KPa]')
                ax_q.grid('on')

                # ax_gama = fig.add_subplot(gs[2,3])
                # ax_gama.plot(t1_vec, np.rad2deg(gama1_sol), label='1st Stage')
                # ax_gama.plot(t2_vec, np.rad2deg(gama2_sol), label='2nd Stage before fairing separation')
                # ax_gama.plot(t3_vec, np.rad2deg(gama3_sol), label='2nd Stage after fairing separation') 
                # ax_gama.set_title('Gama')
                # ax_gama.set_ylabel('Gama [deg]')
                # ax_gama.set_xlabel('Time [s]')
                # ax_gama.grid()
                # ax_gama.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))

                # ax_heat = fig.add_subplot(gs[0,3])
                # # ax_heat.plot(t1_vec, 2*Qdyn1_sol*vel1_sol/1e6.0, label='1st Stage')
                # if RecoveryStrategy != 'EXP':
                #     ax_heat.plot(t4_reentry_vec, hflux4_reentry_sol/1000.0, label='Reentry Burn')
                #     ax_heat.plot(t4_before_landing_vec, hflux4_before_landing_sol/1000.0, label='Landing Burn')

                #     x4_before_landing_intren_sol = x4_after_reentry_scaled_sol
                #     for i in range(N_before_landing_constraints):
                #         x4_before_landing_intren_sol = rocket_return.dynamics_kp1(x4_before_landing_intren_sol, 0, dt4_before_landing_sol/N_before_landing_constraints/rocket_return.scaleT)
                #         x4_before_landing_intren_sol_unscaled = rocket_return.unscale_x(x4_before_landing_intren_sol)
                #         hflux = k_empirical * np.sqrt(atmosphere.rho_fun(rocket_return.local_to_alt(x4_before_landing_intren_sol_unscaled))) * np.linalg.norm(x4_before_landing_intren_sol_unscaled[2:4])**3
                #         ax_heat.plot(t4_reentry_vec[-1] + (i+1)*dt4_before_landing_sol/N_before_landing_constraints, hflux/1000.0, 'ro')

                # ax_heat.set_title('Empirical Heat Flux')
                # ax_heat.set_ylabel('Empirical Heat Flux [KWa/m^2]')
                # ax_heat.grid()


                # plt.tight_layout()

                if Plot_Results == 1: continue

                # Calc parking orbit
                x_parking = Solution['x3'][:,-1].reshape(1,-1)
                x_parking_scaled = LVopt.eci.scale_x(x_parking[0])
                dtheta = 0
                tf = 0
                while True:
                    x_parking_scaled = LVopt.eci.dynamics_kp1(x_parking_scaled, 0, 10.0/LVopt.eci.scaleT)
                    x_parking = np.vstack((x_parking, LVopt.eci.unscale_x(x_parking_scaled).full().flatten()))
                    tf += 10.0
                    dtheta += np.arccos(np.dot(x_parking[-1,0:3],x_parking[-2,0:3])/(np.linalg.norm(x_parking[-1,0:3])*np.linalg.norm(x_parking[-2,0:3])))
                    if dtheta >= 2*np.pi:
                        break

                r_parking = np.linalg.norm(x_parking[:,0:3], axis=1)
                i_apogee = np.argmax(r_parking)
                x_final = x_parking[i_apogee,:].reshape(1,-1)
                x_final[-1,3:6] = Solution['v3_desired'] * x_final[-1,3:6] / np.linalg.norm(x_final[-1,3:6])

                Parking_apogee = np.linalg.norm(x_parking[np.argmax(r_parking),0:3])
                Parking_perigee = np.linalg.norm(x_parking[np.argmin(r_parking),0:3])


                dtheta = 0
                x_final_scaled = LVopt.eci.scale_x(x_final[0])
                while True:
                    x_final_scaled = LVopt.eci.dynamics_kp1(x_final_scaled, 0, 10.0/LVopt.eci.scaleT)
                    x_final = np.vstack((x_final, LVopt.eci.unscale_x(x_final_scaled).full().flatten()))
                    dtheta += np.arccos(np.dot(x_final[-1,0:3],x_final[-2,0:3])/(np.linalg.norm(x_final[-1,0:3])*np.linalg.norm(x_final[-2,0:3])))
                    if dtheta >= 2*np.pi:
                        break
                r_final = np.linalg.norm(x_final[:,0:3], axis=1)

                final_apogee = np.linalg.norm(x_final[np.argmax(r_final),0:3])
                final_perigee = np.linalg.norm(x_final[np.argmin(r_final),0:3])



                phi, theta = np.mgrid[0:np.pi:30j, 0:2*np.pi:30j]
                x = LVopt.eci.R0 * np.sin(phi) * np.cos(theta)/1e3
                y = LVopt.eci.R0 * np.sin(phi) * np.sin(theta)/1e3
                z = LVopt.eci.R0 * np.cos(phi)/1e3

                # Create equatorial plane (z=0)
                theta_plane = np.linspace(0, 2*np.pi, 100)
                r_plane = np.linspace(0, LVopt.eci.R0*1.1, 25)/1000
                r_grid, theta_grid = np.meshgrid(r_plane, theta_plane)
                x_plane = r_grid * np.cos(theta_grid)
                y_plane = r_grid * np.sin(theta_grid)
                z_plane = np.zeros_like(x_plane)

                # Plot Earth
                ax_3d.plot_surface(x, y, z, rstride=2, cstride=2, color='blue', alpha=0.05, edgecolor='none', antialiased=False)

                # Plot equatorial plane
                ax_3d.plot_surface(x_plane, y_plane, z_plane, color='gray', alpha=0.4)

                #plot trajectory
                ax_3d.plot(Solution['x2'][0,:]/1e3, Solution['x2'][1,:]/1e3, Solution['x2'][2,:]/1e3, color=col)
                ax_3d.plot(Solution['x3'][0,:]/1e3, Solution['x3'][1,:]/1e3, Solution['x3'][2,:]/1e3, color=col)
                ax_3d.plot(x_parking[:,0]/1e3, x_parking[:,1]/1e3, x_parking[:,2]/1e3, '--', color=col, label=f'Parking Orbit (apogee={np.round((np.max(r_parking) - LVopt.eci.R0)/1000)}, perigee={np.round((np.min(r_parking) - LVopt.eci.R0)/1000)})')
                ax_3d.plot(x_final[:,0]/1e3, x_final[:,1]/1e3, x_final[:,2]/1e3, color=col, label=f'Final Orbit (apogee={np.round((np.max(final_apogee) - LVopt.eci.R0)/1000)}, perigee={np.round((np.min(final_perigee) - LVopt.eci.R0)/1000)})')
                ax_3d.legend()

                # ECI axes
                max_range = LVopt.eci.R0 * 2/1e3
                ax_3d.quiver(0, 0, 0, max_range, 0, 0, color='r')
                ax_3d.quiver(0, 0, 0, 0, max_range, 0, color='g')
                ax_3d.quiver(0, 0, 0, 0, 0, max_range, color='b')

                ax_3d_xy.plot(Solution['x2'][0,:]/1e3, Solution['x2'][1,:]/1e3, color=col)
                ax_3d_xy.plot(Solution['x3'][0,:]/1e3, Solution['x3'][1,:]/1e3, color=col)
                ax_3d_xy.plot(x_parking[:,0]/1e3, x_parking[:,1]/1e3, '--', color=col)
                ax_3d_xy.plot(x_final[:,0]/1e3, x_final[:,1]/1e3, color=col)
                ax_3d_xy.plot(LVopt.eci.R0 * np.cos(theta[0,:])/1e3, LVopt.eci.R0/1e3 * np.sin(theta[0,:]), color='k', linewidth=3, alpha=0.3)
                ax_3d_xy.set_aspect('equal')
                ax_3d_xy.set_xlabel('X (km)')
                ax_3d_xy.set_ylabel('Y (km)')
                ax_3d_xy.grid('on')
                ax_3d_yz.plot(Solution['x2'][1,:]/1e3, Solution['x2'][2,:]/1e3, color=col)
                ax_3d_yz.plot(Solution['x3'][1,:]/1e3, Solution['x3'][2,:]/1e3, color=col)
                ax_3d_yz.plot(x_parking[:,1]/1e3, x_parking[:,2]/1e3, '--', color=col)
                ax_3d_yz.plot(x_final[:,1]/1e3, x_final[:,2]/1e3, color=col)
                ax_3d_yz.plot(LVopt.eci.R0 * np.cos(theta[0,:])/1e3, LVopt.eci.R0/1e3 * np.sin(theta[0,:]), color='k', linewidth=3, alpha=0.3)
                ax_3d_yz.set_aspect('equal')
                ax_3d_yz.set_xlabel('Y (km)')
                ax_3d_yz.set_ylabel('Z (km)')
                ax_3d_yz.grid('on')
                ax_3d_xz.plot(Solution['x2'][0,:]/1e3, Solution['x2'][2,:]/1e3, color=col)
                ax_3d_xz.plot(Solution['x3'][0,:]/1e3, Solution['x3'][2,:]/1e3, color=col)
                ax_3d_xz.plot(x_parking[:,0]/1e3, x_parking[:,2]/1e3, '--', color=col)
                ax_3d_xz.plot(x_final[:,0]/1e3, x_final[:,2]/1e3, color=col)
                ax_3d_xz.plot(LVopt.eci.R0 * np.cos(theta[0,:])/1e3, LVopt.eci.R0/1e3 * np.sin(theta[0,:]), color='k', linewidth=3, alpha=0.3)
                ax_3d_xz.set_aspect('equal')
                ax_3d_xz.set_xlabel('X (km)')
                ax_3d_xz.set_ylabel('Z (km)')
                ax_3d_xz.grid('on')

                # Match box proportions to the axis spans for equal data units.
                # Older Matplotlib versions only support aspect='auto' in 3D.
                ax_3d.set_box_aspect([
                    np.ptp(ax_3d.get_xlim3d()),
                    np.ptp(ax_3d.get_ylim3d()),
                    np.ptp(ax_3d.get_zlim3d()),
                ])
                ax_3d.set_xlabel('X (km)')
                ax_3d.set_ylabel('Y (km)')
                ax_3d.set_zlabel('Z (km)')
                ax_3d.set_title('Earth in ECI Frame with Equatorial Plane')
                ax_3d.legend()

                plt.tight_layout()



                # ax_traj3d = plt.subplot(1,2,2, projection='3d')
                #     #plot trajectory
                # # ax_traj3d.plot(x1_eci_sol[0,:], x1_eci_sol[1,:], x1_eci_sol[2,:], 'r', label='1st Stage')
                # ax_traj3d.plot(x2_sol[0,:], x2_sol[1,:], x2_sol[2,:], color=col, label='2nd Stage before fairing separation')
                # ax_traj3d.plot(x3_sol[0,:], x3_sol[1,:], x3_sol[2,:], color=col, label='2nd Stage after fairing separation')
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

                icol += 1
                icol %= len(color_vec)

    return result_lines


plt.show()

if __name__ == '__main__':
    RecoveryStrategyVec = ['EXP', 'ASDS', 'RTLS'] # 'EXP', 'ASDS', 'RTLS'
    payload_mass_predefined = -1 # payload mass defined in kg, maximmize the payload mass if set to non-positive value
    Plot_interm = 0
    LV_Configuration = 1 # 1 - Falcon 9, 2 - Starship
    Plot_Results = 0 # 1 - Plot telemetry, 2- Plot 3d trajectory

    ConfigName = 'Falcon9' if LV_Configuration==1 else 'Starship'

    Target_Orbits = []
    ### Target orbit parameters for LEO 
    Target_Orbits.append({"Name": 'LEO',            # Name
                        "apogee": R0 + 200.0e3,     # semi-major axis
                        "perigee": R0 + 200.0e3,    # semi-minor axis
                        "i": np.deg2rad(28.6),})    # inclination [rad]

    ### Target orbit parameters for LEO (ISS)
    # Target_Orbits.append({"Name": 'ISS',                # Name
    #                     "apogee": R0 + 420.0e3,       # semi-major axis
    #                     "perigee": R0 + 415.0e3,      # semi-minor axis
    #                     "i": np.deg2rad(51.6),})       # inclination [rad]
    
    ### Target orbit parameters for SSO
    # Target_Orbits.append({"Name": 'SSO',                # Name
    #                     "apogee": R0 + 780.0e3 ,      # semi-major axis
    #                     "perigee": R0 + 780.0e3,      # semi-minor axis
    #                     "i": np.deg2rad(98.6),})       # inclination [rad]
    
    ### Target orbit parameters for MEO
    Target_Orbits.append({"Name": 'MEO',                # Name
                        "apogee": R0 + 20196.0*1e3,   # semi-major axis
                        "perigee": R0 + 1193.0*1e3,   # semi-minor axis
                        "i": np.deg2rad(55.0),})       # inclination [rad]
    
    ### Target orbit parameters for GTO
    # Target_Orbits.append({"Name": 'GTO',                # Name
    #                     "apogee": R0 + 35786.0e3,       # semi-major axis
    #                     "perigee": R0 + 200.0e3,        # semi-minor axis
    #                     "i": np.deg2rad(28.6),})         # inclination [rad]

    ### Target orbit parameters for TLI
    Target_Orbits.append({"Name": 'TLI',                # Name
                        "apogee": 384400e3,           # apogee [m]
                        "perigee": R0 + 200.0e3,      # perigee [m]
                        "i": np.deg2rad(28.6),})       # inclination [rad]
    
    # Target_Orbits = [Target_Orbits[0]]
    # RecoveryStrategyVec = ['RTLS'] # 'EXP', 'ASDS', 'RTLS'
    
    if Plot_Results > 0:
        plt.ioff()
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        ax_traj = fig.add_subplot(gs[0, 0])
        ax_alt = fig.add_subplot(gs[0,1])
        ax_incl = fig.add_subplot(gs[1,0])
        ax_ap = fig.add_subplot(gs[1,1])
        ax_vel = fig.add_subplot(gs[0,2])
        ax_throttle = fig.add_subplot(gs[1,2])
        ax_mass = fig.add_subplot(gs[2,0])
        ax_acc = fig.add_subplot(gs[1,3])
        ax_q = fig.add_subplot(gs[2,1])
        ax_alpha = fig.add_subplot(gs[2,2])
        ax_Isp = fig.add_subplot(gs[2,3])
    if Plot_Results > 1:
        fig_3d = plt.figure(figsize=(12, 12))
        ax_3d = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3, projection='3d')
        ax_3d_xy = plt.subplot(3,3,3)
        ax_3d_yz = plt.subplot(3,3,6)
        ax_3d_xz = plt.subplot(3,3,9)
    Solution, icol = None, 0
    
    DispesrionList, DispesrionParamList, ConfigList = [], [], []

    DispesrionList.append(np.linspace(-10000,10000, 9))
    DispesrionParamList.append('FirstStage_EmptyMass')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(-3000,10000, 9))
    DispesrionParamList.append('SecondStage_EmptyMass')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(-10000,10000, 9))
    DispesrionParamList.append('FirstStage_EmptyMass')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(-10000,10000, 9))
    DispesrionParamList.append('SecondStage_EmptyMass')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(-10000,10000, 11))
    DispesrionParamList.append('FirstStage_PropMass')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(-10000,10000, 11))
    DispesrionParamList.append('SecondStage_PropMass')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(-10000,10000, 11))
    DispesrionParamList.append('FirstStage_PropMass')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(-10000,10000, 11))
    DispesrionParamList.append('SecondStage_PropMass')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(0.5,2.0, 7))
    DispesrionParamList.append('FirstStageCx0')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(0.5,2.0, 7))
    DispesrionParamList.append('FirstStageCx0')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(0.5,2.0, 7))
    DispesrionParamList.append('BoosterStageCx0')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(0.5,2.0, 7))
    DispesrionParamList.append('BoosterStageCx0')
    ConfigList.append(2)

    ### Engine Isp
    DispesrionList.append(np.linspace(0.85,1.15, 13))
    DispesrionParamList.append('FirstStageIsp')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(0.85,1.15, 13))
    DispesrionParamList.append('SecondStageIsp')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(0.85,1.15, 13))
    DispesrionParamList.append('FirstStageIsp')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(0.85,1.15, 13))
    DispesrionParamList.append('SecondStageIsp')
    ConfigList.append(2)

    ### Engine Thrust
    DispesrionList.append(np.linspace(0.9,1.15, 11))
    DispesrionParamList.append('FirstStageThrust')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(0.85,1.15, 13))
    DispesrionParamList.append('SecondStageThrust')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(0.9,1.15, 11))
    DispesrionParamList.append('FirstStageThrust')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(0.85,1.15, 13))
    DispesrionParamList.append('SecondStageThrust')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(0.0, 5000.0, 6))
    DispesrionParamList.append('LaunchAltDelta')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(0.0, 5000.0, 6))
    DispesrionParamList.append('LaunchAltDelta')
    ConfigList.append(2)

    DispesrionList.append(np.linspace(-0.05, 0.05, 11))
    DispesrionParamList.append('StagePartition')
    ConfigList.append(1)

    DispesrionList.append(np.linspace(-0.05, 0.05, 11))
    DispesrionParamList.append('StagePartition')
    ConfigList.append(2)


    # DispesrionList = np.array([[0]])
    # # RecoveryStrategyVec = RecoveryStrategyVec[2:3]
    # Target_Orbits = Target_Orbits[1:2]
    # DispesrionParamList.append('FirstStage_EmptyMass')
    # ConfigList.append(1)

    EnableParallelDispersion = True
    use_parallel = EnableParallelDispersion and Plot_Results == 0 and len(DispesrionList) > 1
    all_result_lines = []
    if use_parallel:
        workers = min(len(DispesrionList), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result_lines in executor.map(run_dispersion_case, range(len(DispesrionList))):
                all_result_lines.extend(result_lines)
    else:
        for i in range(len(DispesrionList)):
            all_result_lines.extend(run_dispersion_case(i))

    # Print in case order only after every sweep and worker has finished.
    for result_line in all_result_lines:
        print(result_line)

    plt.show()
