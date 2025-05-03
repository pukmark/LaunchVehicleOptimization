#!/usr/bin/env python3

from dataclasses import dataclass, field
import numpy as np
import scipy as sp
import casadi as ca
import Atmosphere_Type as Atm_Type

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
    Diameter: float = field(default = 3.66)

    #Mass Properties
    EmptyFirstStageMass: float = field(default = 20.0 * 10**3)
    EmptySecondStageMass: float = field(default = 4.0 * 10**3)
    # PayloadMass: float = field(default = 22.8 * 10**3)
    FirstStagePropellentMass: float = field(default = 418.7 * 10**3)
    SecondStagePropellentMass: float = field(default = 111.5 * 10**3)
    LV_total_mass: float = field(default = 0.0)
    FirstStage_EmptyMass: float = field(default = 0.0)
    SecondStage_EmptyMass: float = field(default = 0.0)
    SecondStage_FullMass: float = field(default = 0.0)
    FairingMass: float = field(default = 1.9 * 10**3) # [kg]

    #Launch Site - Kennedy Space Center
    LaunchLatitude: float = field(default = 28.6) # [deg]
    LaunchLongitude: float = field(default = -80.6) # [deg]
    # LaunchLatitude: float = field(default = 0.0) # [deg]
    # LaunchLongitude: float = field(default = 0.0) #e[dege]    
    LaunchAltitude: float = field(default = 0.0) # [m]

    #Aerodynamic Properties
    Sref: float = field(default = 0.0)
    FirstStage_CLa: float = field(default = 0.5) # [1/rad]
    FirstStage_Cd0: float = field(default = 0.6)
    FirstStage_Cda2: float = field(default = 3.0)
    FirstStage_MaxAlpha: float = field(default = 0.1745) # [rad]
    FairingSeparationAltitude: float = field(default = 100.0 * 10**3) # [m]
    FirstStage_MaxDynamicPressure: float = field(default = 30.0 * 10**3) # [Pa]

    Booster_Cd0: float = field(default = 1.5)
    Booster_Cda2: float = field(default = 2.0)
    Booster_CLa: float = field(default = 0.6) # [1/rad]
    Booster_MaxDynamicPressure: float = field(default = 100.0 * 10**3) # [Pa]
    Booster_MaxHeatFlux: float = field(default = 300.0 * 10**3) # [Pa]

    #Propulsion Properties
    FirstStage_SL_Isp: float = field(default = 282.0) # [s]
    FirstStage_Vac_Isp: float = field(default = 311.0) # [s]
    FirstStage_SL_Thrust: float = field(default = 845.0 * 10**3 * 9) # [N]
    FirstStage_Vac_Thrust: float = field(default = 914.0* 10**3 * 9) # [N]
    FirstStage_MinThrust_Factor: float = field(default = 0.5) # [-]
    FirstStage_MinThrust_IspFactor: float = field(default = 0.9) # [-]
    FirstStage_Ae: float = field(default = 43.79) # [m^2]
    
    SecondStage_Thrust: float = field(default = 981.0 * 10**3) # [N]
    SecondStage_Vac_Isp: float = field(default = 348.0) # [s]
    SecondStage_MinThrust_Factor: float = field(default = 0.5) # [-]
    SecondStage_MinThrust_IspFactor: float = field(default = 0.95) # [-]

    #Gravity and earth properties
    g0: float = field(default = None)
    R0: float = field(default = 6378137.0) # [m]
    e: float = field(default = 0.0) # [-] earth eccentricity 0.081819190842622
    omega_earth: float = field(default = 7.2921159e-5) # [rad/s] earth rotation rate
    mu: float = field(default = 3.986004418e14) # [m^3/s^2] earth gravitational parameter

    nx: int = field(default=5)
    nu: int = field(default=2)
    dynamics: ca.Function = field(default=None)
    dynamics_kp1: ca.Function = field(default=None)
    dynamics_kp1_M50: ca.Function = field(default=None)
    dynamics_kp1_M20: ca.Function = field(default=None)
    dynamics_kp1_M10: ca.Function = field(default=None)
    local_to_eci_func: ca.Function = field(default=None)

    ISP_calc: ca.Function = field(default=None)

    def __post_init__(self):
        # Mass Properties - without payload
        self.LV_total_mass = self.EmptyFirstStageMass + self.EmptySecondStageMass + self.FirstStagePropellentMass + self.SecondStagePropellentMass + self.FairingMass
        self.FirstStage_EmptyMass = self.LV_total_mass - self.FirstStagePropellentMass
        self.SecondStage_EmptyMass = self.EmptySecondStageMass
        self.SecondStage_FullMass = self.EmptySecondStageMass + self.SecondStagePropellentMass + self.FairingMass
        self.Sref = np.pi * (self.Diameter/2)**2
        self.g0 = self.mu / self.R0**2

    def RK4(self, f, x, u, h, M = 1):
        hM = h/M
        for _ in range(M):
            k1 = f(x, u)
            k2 = f(x + hM/2 * k1, u)
            k3 = f(x + hM/2 * k2, u)
            k4 = f(x + hM * k3, u)
            x = x + hM/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x
    
    def local_to_alt(self, x):
        return np.sqrt(x[0]**2 + (self.R0 + x[1])**2) - self.R0
    
    def eci_to_alt(self, x):
        # local origin in ECI:
        lat = np.deg2rad(self.LaunchLatitude)
        lon = np.deg2rad(self.LaunchLongitude)

        # local origin altitude
        R =  self.R0 / np.sqrt(1 - self.e**2 * np.sin(lat)**2)

        return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - R

@dataclass
class BoosterLaunchVehicle_2D(VLType):
    def __init__(self, atmosphere: Atm_Type.AtmosphereType):
        super().__init__()
        # Mass Properties - without payload
        # self.LV_total_mass = self.EmptyFirstStageMass + self.EmptySecondStageMass + self.FirstStagePropellentMass + self.SecondStagePropellentMass + self.FairingMass
        # self.FirstStage_EmptyMass = self.LV_total_mass - self.FirstStagePropellentMass
        # self.Sref = np.pi * (self.Diameter/2)**2
        # self.g0 = self.mu / self.R0**2

        sym_x = ca.MX.sym('x')
        sym_z = ca.MX.sym('z')
        sym_vx = ca.MX.sym('vx')
        sym_vz = ca.MX.sym('vz')
        sym_m = ca.MX.sym('m')
        sym_dt = ca.MX.sym('dt')

        sym_Tfac = ca.MX.sym('Tfac')
        sym_alpha = ca.MX.sym('alpha')

        sym_q = ca.vertcat(sym_x, sym_z, sym_vx, sym_vz, sym_m)
        sym_u = ca.vertcat(sym_Tfac, sym_alpha)

        self.nx = sym_q.size1()
        self.nu = sym_u.size1()


# define the dynamics function - phase 1 (Booster atmospheric flight)
        gama_v = ca.atan2(sym_vz, sym_vx)
        Alt = ca.norm_2(ca.vertcat(sym_x, self.R0 + sym_z)) - self.R0
        rho = atmosphere.rho_fun(Alt)
        pres = atmosphere.p_fun(Alt)
        sound = atmosphere.a_fun(Alt)
        g = self.g0 * self.R0**2 / (sym_x**2+(self.R0 + sym_z)**2)
        g_angle = sym_x / (self.R0 + sym_z)
        Drag = 0.5 * rho * (self.FirstStage_Cd0 + self.FirstStage_Cda2*sym_alpha**2) * self.Sref * (sym_vx**2 + sym_vz**2)
        Lift = 0.5 * rho * (self.FirstStage_CLa * sym_alpha) * self.Sref * (sym_vx**2 + sym_vz**2)
        Thrust = sym_Tfac**2 * (self.FirstStage_Vac_Thrust - (self.FirstStage_Vac_Thrust-self.FirstStage_SL_Thrust)*pres/atmosphere.p_fun(0))
        Fx = -Drag * ca.cos(gama_v) - Lift * ca.sin(gama_v) + Thrust * ca.cos(sym_alpha + gama_v)
        Fz = -Drag * ca.sin(gama_v) - Lift * ca.cos(gama_v) + Thrust * ca.sin(sym_alpha + gama_v)
        Isp = (self.FirstStage_SL_Isp + (self.FirstStage_Vac_Isp-self.FirstStage_SL_Isp)*pres/atmosphere.p_fun(0)) * (1.0 - (1.0-self.FirstStage_MinThrust_IspFactor) * (1.0-sym_Tfac)/(1.0-self.FirstStage_MinThrust_Factor)  )

        dx = sym_vx
        dz = sym_vz
        dvx = Fx / sym_m - g * ca.sin(g_angle)
        dvz = Fz / sym_m - g * ca.cos(g_angle)
        dm = -Thrust / (Isp * self.g0)

        # define the dynamics function
        dqdt = ca.vertcat(dx, dz, dvx, dvz, dm)
        self.dynamics = ca.Function('dynamics_boost', [sym_q, sym_u], [dqdt])
        self.dynamics_kp1 = ca.Function('dynamics_boost_kp1', [sym_q, sym_u, sym_dt], [self.RK4(self.dynamics, sym_q, sym_u, sym_dt, 1)])
        self.ISP_calc = ca.Function('Isp_calc', [sym_q, sym_u], [Isp])
        
    # def RK4(self, f, x, u, h, M = 1):
    #     hM = h/M
    #     for _ in range(M):
    #         k1 = f(x, u)
    #         k2 = f(x + hM/2 * k1, u)
    #         k3 = f(x + hM/2 * k2, u)
    #         k4 = f(x + hM * k3, u)
    #         x = x + hM/6 * (k1 + 2*k2 + 2*k3 + k4)
    #     return x

@dataclass
class LaunchVehicle_ECI(VLType):
    def __init__(self):
        super().__init__()
        # Mass Properties - without payload
        # self.SecondStage_FullMass = self.EmptySecondStageMass + self.SecondStagePropellentMass + self.FairingMass
        # self.SecondStage_EmptyMass = self.EmptySecondStageMass
        # self.g0 = self.mu / self.R0**2

        # States
        sym_x = ca.MX.sym('x')
        sym_y = ca.MX.sym('y')
        sym_z = ca.MX.sym('z')
        sym_vx = ca.MX.sym('vx')
        sym_vy = ca.MX.sym('vy')
        sym_vz = ca.MX.sym('vz')
        sym_m = ca.MX.sym('m')
        # inputs 
        sym_fx = ca.MX.sym('fx')
        sym_fy = ca.MX.sym('fy')
        sym_fz = ca.MX.sym('fz')
        sym_dt = ca.MX.sym('dt')

        sym_q = ca.vertcat(sym_x, sym_y, sym_z, sym_vx, sym_vy, sym_vz, sym_m)
        sym_u = ca.vertcat(sym_fx, sym_fy, sym_fz)

        self.nx = sym_q.size1()
        self.nu = sym_u.size1()


        # define the dynamics function - phase 2 (exo atmospheric flight)
        g = self.g0 * self.R0**2 / (sym_x**2 + sym_y**2 + sym_z**2)
        Isp = self.SecondStage_Vac_Isp

        dx = sym_vx
        dy = sym_vy
        dz = sym_vz
        dvx = sym_fx / sym_m - g * sym_x / ca.sqrt(sym_x**2 + sym_y**2 + sym_z**2)
        dvy = sym_fy / sym_m - g * sym_y / ca.sqrt(sym_x**2 + sym_y**2 + sym_z**2)
        dvz = sym_fz / sym_m - g * sym_z / ca.sqrt(sym_x**2 + sym_y**2 + sym_z**2)
        dm = -(ca.sqrt(sym_fx**2+sym_fy**2+sym_fz**2)) / (Isp * self.g0)

        # define the dynamics function
        dqdt = ca.vertcat(dx, dy, dz, dvx, dvy, dvz, dm)
        self.dynamics = ca.Function('dynamics_eci', [sym_q, sym_u], [dqdt])
        self.dynamics_kp1 = ca.Function('dynamics_eci_kp1', [sym_q, sym_u, sym_dt], [self.RK4(self.dynamics, sym_q, sym_u, sym_dt, 1)])
        
        self.ISP_calc = ca.Function('Isp_calc', [sym_q, sym_u], [Isp])
        self.local_to_eci()


    
    def local_to_eci(self):

        local_pos_m = ca.MX.sym('local_pos_m', 3)
        local_vel_mps = ca.MX.sym('local_vel_mps', 3)
        az = ca.MX.sym('az')

        # local origin in ECI:
        lat = np.deg2rad(self.LaunchLatitude)
        lon = np.deg2rad(self.LaunchLongitude)

        # local origin altitude
        R =  self.R0 / np.sqrt(1 - self.e**2 * np.sin(lat)**2)

        x = (R + self.LaunchAltitude) * np.cos(lat) * np.cos(lon)
        y = (R + self.LaunchAltitude) * np.cos(lat) * np.sin(lon)
        z = ((1 - self.e**2) * R + self.LaunchAltitude) * np.sin(lat)

        eci_origin =  ca.vertcat(x, y, z)
        # local origin velocity
        v_x = -self.omega_earth * y
        v_y = self.omega_earth * x
        v_z = 0.0
        origin_vel_eci = ca.vertcat(v_x, v_y, v_z)

        # Build local to ECI rotation matrix
        # Local Up (Z)
        z = ca.vertcat(np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))

        # Local North
        n = ca.vertcat(-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat))

        # Local East
        e = ca.vertcat(-np.sin(lon), np.cos(lon), 0.0)

        # X-axis: along azimuth (from North and East)
        x = np.cos(az) * n + np.sin(az) * e

        # Y-axis: complete right-hand rule
        y = ca.vertcat(z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2], z[0]*x[1] - z[1]*x[0])

        # Build rotation matrix
        R_local_to_eci = ca.horzcat(x, y, z)

        # Transform local vectors
        local_pos_eci = R_local_to_eci @ local_pos_m
        local_vel_eci = R_local_to_eci @ local_vel_mps

        # Final ECI state
        pos_eci = eci_origin + local_pos_eci
        vel_eci = origin_vel_eci + local_vel_eci

        self.local_to_eci_func = ca.Function('local_to_eci', [local_pos_m, local_vel_mps, az], [pos_eci, vel_eci])

        return

@dataclass
class BoosterReturn_2D(VLType):
    def __init__(self, atmosphere: Atm_Type.AtmosphereType):
        super().__init__()
        # Mass Properties - without payload
        # self.LV_total_mass = self.EmptyFirstStageMass + self.EmptySecondStageMass + self.FirstStagePropellentMass + self.SecondStagePropellentMass + self.FairingMass
        # self.FirstStage_EmptyMass = self.LV_total_mass - self.FirstStagePropellentMass
        # self.Sref = np.pi * (self.Diameter/2)**2
        # self.g0 = self.mu / self.R0**2

        sym_x = ca.MX.sym('x')
        sym_z = ca.MX.sym('z')
        sym_vx = ca.MX.sym('vx')
        sym_vz = ca.MX.sym('vz')
        sym_m = ca.MX.sym('m')
        sym_dt = ca.MX.sym('dt')

        sym_Tfac = ca.MX.sym('Tfac')
        # sym_alpha = ca.MX.sym('alpha')

        sym_q = ca.vertcat(sym_x, sym_z, sym_vx, sym_vz, sym_m)
        # sym_u = ca.vertcat(sym_Tfac, sym_alpha)
        sym_u = ca.vertcat(sym_Tfac)

        self.nx = sym_q.size1()
        self.nu = sym_u.size1()


# define the dynamics function - phase 1 (Booster atmospheric flight)
        gama_v = ca.atan2(sym_vz, sym_vx)
        Alt = ca.norm_2(ca.vertcat(sym_x, self.R0 + sym_z)) - self.R0
        rho = atmosphere.rho_fun(Alt)
        g = self.g0 * self.R0**2 / (sym_x**2+(self.R0 + sym_z)**2)
        g_angle = ca.asin(sym_x / (self.R0 + sym_z))
        Drag = 0.5 * rho * (self.Booster_Cd0) * self.Sref * (sym_vx**2 + sym_vz**2)
        # Thrust = sym_Tfac * (self.FirstStage_Vac_Thrust - (self.FirstStage_Vac_Thrust-self.FirstStage_SL_Thrust)*pres/atmosphere.p_fun(0))
        Thrust = sym_Tfac * self.FirstStage_SL_Thrust
        Fx = -Drag * ca.cos(gama_v) - Thrust * ca.cos(gama_v)
        Fz = -Drag * ca.sin(gama_v) - Thrust * ca.sin(gama_v)
        Isp = self.FirstStage_SL_Isp

        dx = sym_vx
        dz = sym_vz
        dvx = Fx / sym_m - g * ca.sin(g_angle)
        dvz = Fz / sym_m - g * ca.cos(g_angle)
        dm = -Thrust / (Isp * self.g0)

        # define the dynamics function
        dqdt = ca.vertcat(dx, dz, dvx, dvz, dm)
        self.dynamics = ca.Function('dynamics_boost', [sym_q, sym_u], [dqdt])
        self.dynamics_kp1 = ca.Function('dynamics_boost_kp1', [sym_q, sym_u, sym_dt], [self.RK4(self.dynamics, sym_q, sym_u, sym_dt, 1)])
        self.dynamics_kp1_M50 = ca.Function('dynamics_boost_kp1_M50', [sym_q, sym_u, sym_dt], [self.RK4(self.dynamics, sym_q, sym_u, sym_dt, 50)])
        self.dynamics_kp1_M20 = ca.Function('dynamics_boost_kp1_M50', [sym_q, sym_u, sym_dt], [self.RK4(self.dynamics, sym_q, sym_u, sym_dt, 20)])
        self.dynamics_kp1_M10 = ca.Function('dynamics_boost_kp1_M50', [sym_q, sym_u, sym_dt], [self.RK4(self.dynamics, sym_q, sym_u, sym_dt, 10)])
        self.ISP_calc = ca.Function('Isp_calc', [sym_q, sym_u], [Isp])
        