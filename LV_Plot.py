import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import casadi as ca

import LV_Type_scaled as LV_Type



class LV_plot():
    def __init__(self,  rocket_booster: LV_Type.BoosterLaunchVehicle_2D, 
                        rocket_eci: LV_Type.LaunchVehicle_ECI, 
                        rocket_boostback: LV_Type.BoostBackBurn_2D = None,
                        rocket_return: LV_Type.BoosterReturn_2D = None):
        
        self.rocket_booster = rocket_booster
        self.rocket_eci = rocket_eci
        self.rocket_boostback = rocket_boostback
        self.rocket_return = rocket_return

    def plot_init(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax_vec = []
        for i in range(9):
            self.ax_vec.append(plt.subplot(3,3,i+1))
        self.plot_traj = []
        for i in range(10):
            self.plot_traj.append(self.ax_vec[0].plot([], [])[0])
        self.plot_vel = []
        for i in range(10):
            self.plot_vel.append(self.ax_vec[1].plot([], [])[0])
        self.plot_u = []
        for i in range(10):
            self.plot_u.append(self.ax_vec[2].plot([], [])[0])
        self.plot_mass = []
        for i in range(10):
            self.plot_mass.append(self.ax_vec[3].plot([], [])[0])
        self.plot_alt = []
        for i in range(10):
            self.plot_alt.append(self.ax_vec[4].plot([], [])[0])
        self.plot_inc = []
        for i in range(10):
            self.plot_inc.append(self.ax_vec[5].plot([], [])[0])
        self.plot_Q = []
        for i in range(10):
            self.plot_Q.append(self.ax_vec[6].plot([], [])[0])
        self.plot_pr = []
        for i in range(10):
            self.plot_pr.append(self.ax_vec[7].plot([], [])[0])

        for i in range(9):
            self.ax_vec[i].grid('on')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return

    def plot_close(self):
        plt.close(self.fig)
        self.fig = []
        self.ax_vec = []
        plt.ioff()

        return


    def plot_iteration(self, i, x, Target_Orbit, payload_mass_predefined, optimize_stage_partition:bool = False):

        if i % 3 > 0:
            return

        i0 = 0
        if payload_mass_predefined <= 0.0:
            payload_mass = x[i0] ; i0 += 1
        else:
            payload_mass = payload_mass_predefined/self.rocket_booster.scaleX[4]
        if optimize_stage_partition:
            FirstStage_Propellent_fraction = x[i0] ; i0 += 1
        x1 = x[i0:i0+self.rocket_booster.nx*(self.rocket_booster.N+1)].reshape(self.rocket_booster.N+1, self.rocket_booster.nx) ; i0 += self.rocket_booster.nx*(self.rocket_booster.N+1)
        u1 = x[i0:i0+self.rocket_booster.nu*self.rocket_booster.N].reshape(self.rocket_booster.N, self.rocket_booster.nu) ; i0 += self.rocket_booster.nu*self.rocket_booster.N
        dt1 = x[i0] * self.rocket_booster.scaleT; i0 += 1
        LaunchAz = x[i0]; i0 += 1
        x21 = x[i0:i0+self.rocket_eci.nx*(self.rocket_eci.N[0]+1)].reshape(self.rocket_eci.N[0]+1, self.rocket_eci.nx); i0 += self.rocket_eci.nx*(self.rocket_eci.N[0]+1)
        u21 = x[i0:i0+self.rocket_eci.nu*self.rocket_eci.N[0]].reshape(self.rocket_eci.N[0], self.rocket_eci.nu); i0 += self.rocket_eci.nu*self.rocket_eci.N[0]
        dt21 = x[i0] * self.rocket_eci.scaleT; i0 += 1
        x22 = x[i0:i0+self.rocket_eci.nx*(self.rocket_eci.N[1]+1)].reshape(self.rocket_eci.N[1]+1, self.rocket_eci.nx); i0 += self.rocket_eci.nx*(self.rocket_eci.N[1]+1)
        u22 = x[i0:i0+self.rocket_eci.nu*self.rocket_eci.N[1]].reshape(self.rocket_eci.N[1], self.rocket_eci.nu); i0 += self.rocket_eci.nu*self.rocket_eci.N[1]
        dt22 = x[i0] * self.rocket_eci.scaleT; i0 += 1
        x2 = np.vstack((x21[:-1,:],x22))
        u2 = np.vstack((u21,u22))
        if self.rocket_boostback is not None:
            dt4_boostback = x[i0] * self.rocket_boostback.scaleT; i0 += 1
            x4_boostback = x[i0:i0+self.rocket_boostback.nx]; i0 += self.rocket_boostback.nx
            u4_boostback = x[i0:i0+self.rocket_boostback.nu]; i0 += self.rocket_boostback.nu
        else:
            dt4_boostback = 0.0
            
        if self.rocket_return is not None:
            x4_0 = x[i0:i0+self.rocket_return.nx]; i0 += self.rocket_return.nx
            x4_before_reentry = x[i0:i0+self.rocket_return.nx]; i0+=self.rocket_return.nx
            x4_after_reentryburn = x[i0:i0+self.rocket_return.nx]; i0+=self.rocket_return.nx
            u4_reentry = x[i0:i0+self.rocket_return.nu]; i0+=self.rocket_return.nu
            x4_before_landing = x[i0:i0+self.rocket_return.nx]; i0+=self.rocket_return.nx
            x4_landing = x[i0:i0+self.rocket_return.nx*(self.rocket_return.N+1)].reshape(self.rocket_return.N+1, self.rocket_return.nx); i0+=self.rocket_return.nx*(self.rocket_return.N+1)
            u4_landing = x[i0:i0+self.rocket_return.nu*self.rocket_return.N].reshape(self.rocket_return.N, self.rocket_return.nu); i0+=self.rocket_return.nu*self.rocket_return.N
            dt4_reentry = x[i0] * self.rocket_return.scaleT; i0+=1
            dt4_ballistic = x[i0] * self.rocket_return.scaleT; i0+=1
            dt4_before_landing = x[i0] * self.rocket_return.scaleT; i0+=1
            dt4_landing = x[i0] * self.rocket_return.scaleT; i0+=1
        else:
            t4_landing = np.array([0,0])

        t1_vec = np.linspace(0.0, dt1*self.rocket_booster.N, self.rocket_booster.N+1)
        t21_start = t1_vec[-1] + self.rocket_eci.SecondStage_CoastTimeAfterSep
        t21_vec = t21_start + np.linspace(0.0, dt21*self.rocket_eci.N[0], self.rocket_eci.N[0]+1).reshape(-1,1)
        t22_vec = t21_vec[-1] + np.linspace(0.0, dt22*self.rocket_eci.N[1], self.rocket_eci.N[1]+1).reshape(-1,1)
        t2_vec = np.vstack((t21_vec[:-1,:],t22_vec)).reshape(-1)

        x1p = np.zeros_like(x1)
        alt1, alt2 = np.zeros((x1.shape[0])), np.zeros((x2.shape[0]))
        Qdyn1 = np.zeros((x1.shape[0]))
        for k in range(x1.shape[0]):
            x1p[k,:] = self.rocket_booster.unscale_x(x1[k,:]).full().flatten()
            alt1[k] = self.rocket_booster.local_to_alt(x1p[k,:])
            Qdyn1[k] = self.rocket_booster.dynamic_pressure_fun(x1[k,:])
        x2p = np.zeros_like(x2)
        for k in range(x2.shape[0]):
            x2p[k,:] = self.rocket_eci.unscale_x(x2[k,:]).full().flatten()
            alt2[k] = self.rocket_eci.eci_to_alt(x2p[k,:])
        if self.rocket_return is not None:
            N4 = self.rocket_return.N
            x4p = np.zeros((3*N4+self.rocket_return.N+1,self.rocket_return.nx))
            t4_vec = np.zeros((3*N4+self.rocket_return.N+1,))
            i = 0
            if self.rocket_boostback is not None:
                x4p = np.vstack((x4p,np.zeros((N4,self.rocket_return.nx))))
                t4_vec = np.vstack((t4_vec.reshape(-1,1), np.zeros((N4,1)))).reshape(-1)
                x4p[0,:] = self.rocket_boostback.unscale_x(x4_0).full().flatten()
                t4_vec[0] = t1_vec[-1]
                for k in range(self.rocket_return.N):
                    t4_vec[i+1] = t4_vec[i] + dt4_boostback/N4
                    x4p[i+1,:] = self.rocket_boostback.unscale_x(self.rocket_boostback.dynamics_kp1(self.rocket_boostback.scale_x(x4p[i,:]), u4_boostback, self.rocket_boostback.scale_t(dt4_boostback/self.rocket_return.N))).full().flatten()
                    i += 1
            else:
                t4_vec[0] = t1_vec[-1]
                x4p[0,:] = self.rocket_return.unscale_x(x4_0).full().flatten()
            for k in range(0,N4):
                t4_vec[i+1] = t4_vec[i] + dt4_ballistic/N4
                x4p[i+1,:] = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1(self.rocket_return.scale_x(x4p[i,:]), 0, self.rocket_return.scale_t(dt4_ballistic/N4))).full().flatten()
                i += 1
            for k in range(N4):
                t4_vec[i+1] = t4_vec[i] + dt4_reentry/N4
                x4p[i+1,:] = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1(self.rocket_return.scale_x(x4p[i,:]), u4_reentry, self.rocket_return.scale_t(dt4_reentry/N4))).full().flatten()
                i += 1
            for k in range(N4):
                t4_vec[i+1] = t4_vec[i] + dt4_before_landing/N4
                x4p[i+1,:] = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1(self.rocket_return.scale_x(x4p[i,:]), 0, self.rocket_return.scale_t(dt4_before_landing/N4))).full().flatten()
                i += 1
            for k in range(self.rocket_return.N):
                t4_vec[i+1] = t4_vec[i] + dt4_landing
                x4p[i+1,:] = self.rocket_return.unscale_x(self.rocket_return.dynamics_kp1(self.rocket_return.scale_x(x4p[i,:]), u4_landing[k,0], self.rocket_return.scale_t(dt4_landing))).full().flatten()
                i += 1
            alt4, Qdyn4 = np.zeros((x4p.shape[0])), np.zeros((x4p.shape[0]))
            for k in range(x4p.shape[0]):
                alt4[k] = self.rocket_return.local_to_alt(x4p[k,:])
                Qdyn4[k] = self.rocket_return.dynamic_pressure_fun(self.rocket_return.scale_x(x4p[k,:]))

        self.plot_traj[0].set_data(x1p[:,0]/1000, x1p[:,1]/1000)
        # if self.rocket_return is not None:
            # self.plot_traj[1].set_data(x4_landing[:,0] * self.rocket_return.scaleX[0]/1000, x4_landing[:,1] * self.rocket_return.scaleX[1]/1000)
        if self.rocket_boostback is not None:
            x_vec = np.vstack([x4_before_reentry[0], x4_after_reentryburn[0], x4_before_landing[0],x4_landing[:,0].reshape(-1,1)])
            z_vec = np.vstack([x4_before_reentry[1], x4_after_reentryburn[1], x4_before_landing[1],x4_landing[:,1].reshape(-1,1)])
            self.plot_traj[1].set_data(np.array([x4_0[0], x4_boostback[0]]) * self.rocket_boostback.scaleX[0]/1000, np.array([x4_0[1], x4_boostback[1]]) * self.rocket_boostback.scaleX[1]/1000)
            self.plot_traj[2].set_data(x_vec * self.rocket_return.scaleX[0]/1000, z_vec * self.rocket_return.scaleX[1]/1000)
            self.plot_traj[1].set_linestyle(''); self.plot_traj[1].set_marker('s')
            self.plot_traj[2].set_linestyle(''); self.plot_traj[1].set_marker('s')
        elif self.rocket_return is not None:
            x_vec = np.vstack([x4_0[0], x4_before_reentry[0], x4_after_reentryburn[0], x4_before_landing[0],x4_landing[:,0].reshape(-1,1)])
            z_vec = np.vstack([x4_0[1], x4_before_reentry[1], x4_after_reentryburn[1], x4_before_landing[1],x4_landing[:,1].reshape(-1,1)])
            self.plot_traj[1].set_data(x_vec * self.rocket_return.scaleX[0]/1000, z_vec * self.rocket_return.scaleX[1]/1000)
            self.plot_traj[1].set_linestyle(''); self.plot_traj[1].set_marker('s')
            
            if np.max(x_vec* self.rocket_return.scaleX[0]) > 200e3:
                x_earth = np.linspace(0, 850000,1000)
                z_earth = np.sqrt(self.rocket_return.R0**2 - x_earth**2) - self.rocket_return.R0
                self.plot_traj[2].set_data(x_earth/1000, z_earth/1000)
                self.plot_traj[2].set_linestyle('--')
        if self.rocket_return is not None:
            self.plot_traj[3].set_data(x4p[:,0]/1000, x4p[:,1]/1000)


        self.plot_vel[0].set_data(t1_vec, np.linalg.norm(x1p[:,2:4], axis=1))
        self.plot_vel[1].set_data(t2_vec, np.linalg.norm(x2p[:,3:6], axis=1))
        if self.rocket_boostback is not None:
            self.plot_vel[3].set_data([t1_vec[-1]+dt4_boostback], [np.linalg.norm(x4_boostback[2:4])* self.rocket_boostback.scaleX[2]])
        if self.rocket_return is not None:
            self.plot_vel[4].set_data([t1_vec[-1]+dt4_boostback+dt4_ballistic], [np.linalg.norm(x4_before_reentry[2:4])* self.rocket_return.scaleX[2]])
            self.plot_vel[5].set_data([t1_vec[-1]+dt4_boostback+dt4_ballistic+dt4_reentry], [np.linalg.norm(x4_after_reentryburn[2:4])* self.rocket_return.scaleX[2]])
            self.plot_vel[6].set_data([t1_vec[-1]+dt4_boostback+dt4_ballistic+dt4_reentry+dt4_before_landing], [np.linalg.norm(x4_before_landing[2:4])* self.rocket_return.scaleX[2]])
            t4_reentry = t2_vec[-1]+dt4_boostback+dt4_ballistic
            t4_landing = t2_vec[-1]+dt4_boostback+dt4_ballistic+dt4_reentry+dt4_before_landing + np.linspace(0,dt4_landing*self.rocket_return.N, self.rocket_return.N+1)
            self.plot_vel[7].set_data(t4_vec, np.linalg.norm(x4p[:,2:4], axis=1))
            self.plot_vel[3].set_marker('s'); self.plot_vel[3].set_linestyle('')
            self.plot_vel[4].set_marker('s'); self.plot_vel[3].set_linestyle('')
            self.plot_vel[5].set_marker('s'); self.plot_vel[3].set_linestyle('')
            self.plot_vel[6].set_marker('s'); self.plot_vel[3].set_linestyle('')
            
        self.plot_u[0].set_data(t1_vec[:-1], u1[:,0])
        self.plot_u[1].set_data(t2_vec[:-1], np.linalg.norm(u2, axis=1))
        if self.rocket_return is not None:
            self.plot_u[3].set_data(t4_landing[:-1], u4_landing)
            self.plot_u[4].set_data([t4_reentry, t4_reentry+dt4_reentry], [u4_reentry, u4_reentry])
        self.plot_u[5].set_data(t1_vec[:-1], u1[:,1])

        self.plot_mass[0].set_data(t1_vec, x1p[:,4]/1000)
        self.plot_mass[1].set_data(t2_vec, x2p[:,6]/1000)
        if self.rocket_return is not None:
            self.plot_mass[2].set_data(t4_vec, x4p[:,4]/1000)

        self.plot_alt[0].set_data(t1_vec, alt1/1000)
        self.plot_alt[1].set_data(t2_vec, alt2/1000)
        if self.rocket_return is not None:
            self.plot_alt[2].set_data(t4_vec, alt4/1000)
        
        i2, apogee2, perigee2 = np.zeros((x2.shape[0],)), np.zeros((x2.shape[0],)), np.zeros((x2.shape[0],))
        for k in range(x2.shape[0]):
            h = ca.cross(x2p[k,0:3], x2p[k,3:6])
            i2[k] = np.arccos(h[2] / ca.norm_2(h))
            e = ca.cross(x2p[k,3:6], h) / (self.rocket_eci.mu) - x2p[k,0:3] / np.linalg.norm(x2p[k,0:3])
            eps = 0.5*ca.sumsqr(x2p[k,3:6]) - self.rocket_eci.mu / np.linalg.norm(x2p[k,0:3])
            a = -self.rocket_eci.mu / (2.0*eps)
            apogee2[k] = a * (1.0 + np.linalg.norm(e))
            perigee2[k] = a * (1.0 - np.linalg.norm(e))

        self.plot_inc[0].set_data(t2_vec, np.rad2deg(i2))
        self.plot_inc[2].set_data([0, t2_vec[-1]+10], [np.rad2deg(Target_Orbit['i']), np.rad2deg(Target_Orbit['i'])])
        self.plot_inc[2].set_linestyle('--')

        self.plot_pr[0].set_data(t2_vec, (apogee2 - self.rocket_eci.R0)/1000)
        self.plot_pr[1].set_data(t2_vec, (perigee2- self.rocket_eci.R0)/1000)
        self.plot_pr[2].set_data([0, t2_vec[-1]+10], [(Target_Orbit['apogee']- self.rocket_eci.R0)/1000, (Target_Orbit['apogee']- self.rocket_eci.R0)/1000])
        self.plot_pr[2].set_linestyle('--')
        self.plot_pr[3].set_data([0, t2_vec[-1]+10], [(Target_Orbit['perigee']- self.rocket_eci.R0)/1000, (Target_Orbit['perigee']- self.rocket_eci.R0)/1000])
        self.plot_pr[3].set_linestyle('--')
        self.plot_pr[4].set_data([0, t2_vec[-1]+10], [self.rocket_eci.ParkingOrbit_PerigeeAlt/1000, self.rocket_eci.ParkingOrbit_PerigeeAlt/1000])
        self.plot_pr[4].set_linestyle('--')

        self.plot_Q[0].set_data(t1_vec, Qdyn1/1000)
        if self.rocket_return is not None:
            self.plot_Q[1].set_data(t4_vec, Qdyn4/1000)


        if self.rocket_return is None:
            self.ax_vec[0].set_xlim([-10, x1p[-1,0]/1000+10])
            self.ax_vec[0].set_ylim([0, x1p[-1,1]/1000+10])
        else:
            self.ax_vec[0].set_xlim([-10, max(x1p[-1,0]/1000, np.max(x4p[:,0])/1000)+10])
            self.ax_vec[0].set_ylim([-30, max(x1p[-1,1]/1000, np.max(x4p[:,1])/1000)+10])
        self.ax_vec[0].set_title(f"payload mass = {payload_mass * self.rocket_booster.scaleX[4]:.2f}, LaunchAz = {LaunchAz*180/np.pi:.2f}")
        self.ax_vec[0].set_aspect('equal')
        self.ax_vec[1].set_ylim([0, 8000])
        self.ax_vec[1].set_xlim([0, max(t4_landing[-1],t2_vec[-1])+100])
        self.ax_vec[2].set_ylim([-1, 1.5])
        self.ax_vec[2].set_xlim([0, max(t4_landing[-1],t2_vec[-1])+100])
        if optimize_stage_partition:
            self.ax_vec[3].set_title(f'First Stage Propellent Mass Fraction - {FirstStage_Propellent_fraction:2.4}')
        self.ax_vec[3].set_ylim([0, self.rocket_booster.LV_total_mass/1000])
        self.ax_vec[3].set_xlim([0, max(t4_landing[-1],t2_vec[-1])+100])
        self.ax_vec[4].set_ylim([-10, 1.1/1000*max(np.max(alt1), np.max(alt2))])
        self.ax_vec[4].set_xlim([0, t2_vec[-1]+10])
        self.ax_vec[5].set_ylim([-1.1*np.rad2deg(Target_Orbit["i"]), 1.1*np.rad2deg(Target_Orbit["i"])])
        self.ax_vec[5].set_xlim([0, t2_vec[-1]+10])
        if self.rocket_return is not None:
            self.ax_vec[6].set_ylim([0, 1.1/1000*max(np.max(Qdyn1), np.max(Qdyn4))])
            self.ax_vec[6].set_xlim([0, t4_vec[-1]+10])
        else:
            self.ax_vec[6].set_ylim([0, 1.1/1000*np.max(Qdyn1)])
            self.ax_vec[6].set_xlim([0, t1_vec[-1]+10])
        self.ax_vec[7].set_ylim([0, 1.1/1000*(max(np.max(perigee2), np.max(apogee2))-self.rocket_eci.R0)])
        self.ax_vec[7].set_xlim([0, t2_vec[-1]+10])
        self.ax_vec[7].set_yscale('symlog', linthresh=100)


        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return
