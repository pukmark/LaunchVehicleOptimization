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
class DispesrionFactorsType(PythonMsg):
    '''
    base class for creating types and messages in python
    '''
    FirstStageIsp: int = field(default = 1.0)
    SecondStageIsp: int = field(default = 1.0)
    FirstStageThrust: int = field(default = 1.0)
    SecondStageThrust: int = field(default = 1.0)
    FirstStageCx0: int = field(default = 1.0)
    BoosterStageCx0: int = field(default = 1.0)
    FirstStageDeltaInertMass: int = field(default = 0.0)
    SecondStageDeltaInertMass: int = field(default = 0.0)
    AtmosphereDensity: int = field(default = 1.0)
    LaunchAltDelta: int = field(default = 0.0)

@dataclass
class VLType(PythonMsg):
    '''
    base class for creating types and messages in python
    '''
    N: int = field(default = 60)
    Diameter: float = field(default = 3.66)

    #Mass Properties
    EmptyFirstStageMass: float = field(default = 25.6 * 10**3)
    EmptySecondStageMass: float = field(default = 4.0 * 10**3)
    # PayloadMass: float = field(default = 22.8 * 10**3)
    FirstStagePropellentMass: float = field(default = 395.7 * 10**3)
    SecondStagePropellentMass: float = field(default = 92.670 * 10**3)
    LV_total_mass: float = field(default = 0.0)
    FirstStage_EmptyMass: float = field(default = 0.0)
    SecondStage_EmptyMass: float = field(default = 0.0)
    SecondStage_FullMass: float = field(default = 0.0)
    FairingMass: float = field(default = 1.9 * 10**3) # [kg]
    TotalPropellentMass: float = field(default = 0.0) # [kg]
    
    #Launch Site - Kennedy Space Center
    LaunchLatitude: float = field(default = 28.6) # [deg]
    LaunchLongitude: float = field(default = -80.6) # [deg]
    LaunchAltitude: float = field(default = 0.0) # [m]
    atmosphere: Atm_Type.AtmosphereType = field(default=None)

    #Aerodynamic Properties
    Sref: float = field(default = 0.0)
    FirstStage_CLa: float = field(default = 0.5) # [1/rad]
    FirstStage_Cd0: float = field(default = 1.0)
    FirstStage_Cda2: float = field(default = 3.0)
    FirstStage_MaxAlpha: float = field(default = 0.1745) # [rad]
    FairingSeparationAltitude: float = field(default = 100.0 * 10**3) # [m]
    FirstStage_MaxDynamicPressure: float = field(default = 30.0 * 10**3) # [Pa]
    FirstStage_StageSeparationMaxDynamicPressure: float = field(default = 0.5 * 10**3) # [Pa]

    Booster_Cd0: float = field(default = 1.3)
    Booster_Cda2: float = field(default = 2.0)
    Booster_CLa: float = field(default = 0.6) # [1/rad]
    Booster_MaxDynamicPressure: float = field(default = 100.0 * 10**3) # [Pa]
    Booster_MaxHeatFlux: float = field(default = 100.0 * 10**3) # [Pa]
    Booster_k_empirical: float = field(default = 2.0e-4) # [-]

    #Propulsion Properties
    FirstStage_SL_Isp: float = field(default = 282.0) # [s]
    FirstStage_Vac_Isp: float = field(default = 310.0) # [s]
    FirstStage_SL_Thrust: float = field(default = 845.0 * 10**3 * 9) # [N]
    FirstStage_Vac_Thrust: float = field(default = 914.0* 10**3 * 9) # [N]
    FirstStage_MinThrust_Factor: float = field(default = 0.5) # [-]
    FirstStage_MinThrust_IspFactor: float = field(default = 0.975) # [-]
    FirstStage_Ae: float = field(default = 43.79) # [m^2]
    
    SecondStage_Thrust: float = field(default = 981.0 * 10**3) # [N]
    SecondStage_Vac_Isp: float = field(default = 348.0) # [s]
    SecondStage_MinThrust_Factor: float = field(default = 0.5) # [-]
    SecondStage_MinThrust_IspFactor: float = field(default = 0.975) # [-]

    Payload_Max_acc: float = field(default = 4.0*9.81) # [m/s^2]

    #Gravity and earth properties
    g0: float = field(default = None)
    R0: float = field(default = 6378137.0) # [m]
    e: float = field(default = 0.0) # [-] earth eccentricity 0.081819190842622
    omega_earth: float = field(default = 7.2921159e-5) # [rad/s] earth rotation rate
    mu: float = field(default = 3.986004418e14) # [m^3/s^2] earth gravitational parameter

    # Parking Orbit:
    ParkingOrbit_PerigeeAlt: float = field(default = 150.0 * 10**3) # [m^3/s^2] earth gravitational parameter

    nx: int = field(default=5)
    nu: int = field(default=2)
    dynamics: ca.Function = field(default=None)
    dynamics_kp1: ca.Function = field(default=None)
    dynamics_kp1_M50: ca.Function = field(default=None)
    dynamics_kp1_M20: ca.Function = field(default=None)
    dynamics_kp1_M10: ca.Function = field(default=None)
    local_to_eci_func: ca.Function = field(default=None)
    specific_acc_fun: ca.Function = field(default=None)
    dynamic_pressure_fun: ca.Function = field(default=None)
    heat_flux_fun: ca.Function = field(default=None)

    ISP_calc: ca.Function = field(default=None)

    scaleX: list = field(default=None)
    scaleU: list = field(default=None)
    scaleT: list = field(default=None)

    def __post_init__(self):
        # Mass Properties - without payload
        self.LV_total_mass = self.EmptyFirstStageMass + self.EmptySecondStageMass + self.FirstStagePropellentMass + self.SecondStagePropellentMass + self.FairingMass
        self.FirstStage_EmptyMass = self.LV_total_mass - self.FirstStagePropellentMass
        self.SecondStage_FullMass = self.EmptySecondStageMass + self.SecondStagePropellentMass + self.FairingMass
        self.SecondStage_EmptyMass = self.EmptySecondStageMass
        self.TotalPropellentMass = self.FirstStagePropellentMass + self.SecondStagePropellentMass
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
    def scale_x(self, x):
        return ca.vertcat(*[x[i] / self.scaleX[i] for i in range(len(self.scaleX))])
    def scale_u(self, u):
        return ca.vertcat(*[u[i] / self.scaleU[i] for i in range(len(self.scaleU))])
    def scale_t(self, t):
        return t / self.scaleT
    def unscale_x(self, x):
        return ca.vertcat(*[x[i] * self.scaleX[i] for i in range(len(self.scaleX))])
    def unscale_u(self, u):
        return ca.vertcat(*[u[i] * self.scaleU[i] for i in range(len(self.scaleU))])    
    def unscale_t(self, t):
        return t * self.scaleT
    
    def StarShipDatabase(self):
        self.Diameter = 9.0  # Starship outer diameter in meters

        # Mass Properties
        self.EmptyFirstStageMass = 275.0 * 10**3      # Super Heavy dry mass [kg]
        self.EmptySecondStageMass = 85.0 * 10**3      # Starship (upper stage) dry mass [kg]
        self.FirstStagePropellentMass = 3250.0 * 10**3 # Super Heavy propellant [kg]
        self.SecondStagePropellentMass = 1500.0 * 10**3# Starship propellant [kg]
        self.FairingMass = 0.0

        # Aerodynamic Properties
        self.FirstStage_CLa = 0.4  # Less lift from cylindrical shape
        self.FirstStage_Cd0 = 0.8  # More drag due to wider diameter
        self.FirstStage_Cda2 = 3.5
        self.FirstStage_MaxAlpha = 0.1745  # [rad]
        self.FirstStage_MaxDynamicPressure = 25.0 * 10**3  # [Pa]
        self.FirstStage_StageSeparationMaxDynamicPressure = 5.0 * 10**3  # [Pa]

        self.Booster_Cd0 = 1.1  # with grid fins
        self.Booster_Cda2 = 1.8
        self.Booster_CLa = 0.5
        self.Booster_MaxDynamicPressure = 120.0 * 10**3  # [Pa]
        self.Booster_MaxHeatFlux = 150.0 * 10**3  # [W/m^2]
        self.Booster_k_empirical = 2.0e-4

        # Propulsion Properties
        self.FirstStage_SL_Isp = 327.0  # Raptor 2 sea-level [s]
        self.FirstStage_Vac_Isp = 347.0  # Raptor 2 vacuum [s]
        self.FirstStage_SL_Thrust = 2256.3 * 10**3 * 33  # ~33 Raptors on Super Heavy [N]
        self.FirstStage_Vac_Thrust = 2394.3 * 10**3 * 33  # estimated [N]
        self.FirstStage_MinThrust_Factor = 0.5  # [-]
        self.FirstStage_MinThrust_IspFactor = 0.85
        self.FirstStage_Ae = 43.8  # est. total Ae for all engines [m^2]
        self.FairingSeparationAltitude = 80.0 * 10**3 # [m]

        self.SecondStage_Thrust = 2300.0 * 10**3 * 3 + 2256.3 * 10**3 * 3  # 3 vacuum and 3 SL variants [N]
        self.SecondStage_Vac_Isp = (3*380.0 + 3*347.0)/6
        self.SecondStage_MinThrust_Factor = 0.5
        self.SecondStage_MinThrust_IspFactor = 0.85

        self.__post_init__()

    def ApplyScenarioDispersionToLV(self, DispesrionFactors: DispesrionFactorsType):

        self.FirstStage_SL_Isp *= DispesrionFactors.FirstStageIsp
        self.FirstStage_Vac_Isp *= DispesrionFactors.FirstStageIsp
        self.SecondStage_Vac_Isp *= DispesrionFactors.SecondStageIsp
        self.FirstStage_SL_Thrust *= DispesrionFactors.FirstStageThrust
        self.FirstStage_Vac_Thrust *= DispesrionFactors.FirstStageThrust
        self.SecondStage_Thrust *= DispesrionFactors.SecondStageThrust        
        self.FirstStage_Cd0 *= DispesrionFactors.FirstStageCx0
        self.Booster_Cd0 *= DispesrionFactors.BoosterStageCx0
        self.FirstStage_EmptyMass += DispesrionFactors.FirstStageDeltaInertMass
        self.SecondStage_EmptyMass += DispesrionFactors.SecondStageDeltaInertMass
        self.LaunchAltitude += DispesrionFactors.LaunchAltDelta

        self.__post_init__()
        
@dataclass
class BoosterLaunchVehicle_2D(VLType):
    def __init__(self, atmosphere: Atm_Type.AtmosphereType,
                       LV_Configuration: int, 
                       ScenarioDispersion: DispesrionFactorsType, 
                       N: int = 60):
        super().__init__()
        self.N = N
        self.atmosphere = atmosphere
        if LV_Configuration == 2: # Starship database
            self.StarShipDatabase()

        self.ApplyScenarioDispersionToLV(ScenarioDispersion)
        
        self.scaleX = [1e5, 1e5, 1e3, 1e3, 1e5]
        self.scaleU = [1.0, self.FirstStage_MaxAlpha] 
        self.scaleT = 100.0

        sx = ca.MX.sym('sx')
        sz = ca.MX.sym('sz')
        svx = ca.MX.sym('svx')
        svz = ca.MX.sym('svz')
        sm = ca.MX.sym('sm')
        sTfac = ca.MX.sym('sTfac')
        salpha = ca.MX.sym('salpha')
        sdt = ca.MX.sym('sdt')

        sym_q = ca.vertcat(sx, sz, svx, svz, sm)
        sym_u = ca.vertcat(sTfac, salpha)

        self.nx = sym_q.size1()
        self.nu = sym_u.size1()

        x = sx * self.scaleX[0]
        z = sz * self.scaleX[1]
        vx = svx * self.scaleX[2]
        vz = svz * self.scaleX[3]
        m = sm * self.scaleX[4]
        Tfac = sTfac * self.scaleU[0]
        alpha = salpha * self.scaleU[1]

        Alt = self.local_to_alt(ca.vertcat(x, z, vx, vz, m))
        rho = atmosphere.rho_fun(Alt) * ScenarioDispersion.AtmosphereDensity
        pres = atmosphere.p_fun(Alt)
        g = self.g0 * self.R0 ** 2 / (x**2 + (self.R0 + z) ** 2)
        g_angle = ca.asin(x / (self.R0 + z))
        v2 = vx**2 + vz**2

        Drag = 0.5 * rho * v2 * self.Sref * (self.FirstStage_Cd0*ScenarioDispersion.FirstStageCx0 + self.FirstStage_Cda2 * alpha ** 2)
        Lift = 0.5 * rho * v2 * self.Sref * (self.FirstStage_CLa * alpha)
        gama_v = ca.atan2(vz, vx)

        Thrust = ScenarioDispersion.FirstStageThrust * Tfac * (self.FirstStage_Vac_Thrust - (self.FirstStage_Vac_Thrust - self.FirstStage_SL_Thrust) * pres / atmosphere.p_fun(0))
        Isp = ScenarioDispersion.FirstStageIsp * (self.FirstStage_Vac_Isp - (self.FirstStage_Vac_Isp - self.FirstStage_SL_Isp) * pres / atmosphere.p_fun(0)) * (1.0 - (1.0 - self.FirstStage_MinThrust_IspFactor) * (1.0 - Tfac) / (1.0 - self.FirstStage_MinThrust_Factor))

        Fx = -Drag * ca.cos(gama_v) - Lift * ca.sin(gama_v) + Thrust * ca.cos(alpha + gama_v)
        Fz = -Drag * ca.sin(gama_v) - Lift * ca.cos(gama_v) + Thrust * ca.sin(alpha + gama_v)

        dx = vx
        dz = vz
        dvx = Fx / m - g * ca.sin(g_angle)
        dvz = Fz / m - g * ca.cos(g_angle)
        # dm = -Thrust / (Isp * self.g0)
        # dm = -self.FirstStage_Vac_Thrust / (self.FirstStage_Vac_Isp * self.g0) * Tfac / (1.0 - (1.0 - self.FirstStage_MinThrust_IspFactor) * (1.0 - Tfac) / (1.0 - self.FirstStage_MinThrust_Factor))
        dm = -self.FirstStage_Vac_Thrust / (self.FirstStage_Vac_Isp * self.g0) * Tfac 

        dqdt_scaled = ca.vertcat(dx / self.scaleX[0], dz / self.scaleX[1],
            dvx / self.scaleX[2], dvz / self.scaleX[3],
            dm / self.scaleX[4]) * self.scaleT

        sym_sx = ca.vertcat(sx, sz, svx, svz, sm)
        sym_su = ca.vertcat(sTfac, salpha)
        self.dynamics = ca.Function('dynamics_scaled', [sym_sx, sym_su], [dqdt_scaled])
        self.dynamics_kp1 = ca.Function('dynamics_kp1_scaled', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt)])
        self.ISP_calc = ca.Function('Isp_calc_scaled', [sym_sx, sym_su], [Isp])
        self.specific_acc_fun = ca.Function('Isp_calc_scaled', [sym_sx, sym_su], [ca.sqrt(Fx**2+Fz**2)/m])
        self.dynamic_pressure_fun = ca.Function('dynamic_pressure_scaled', [sym_sx], [0.5 * rho * v2])
        self.heat_flux_fun = ca.Function('heat_flux_scaled', [sym_sx], [self.Booster_k_empirical * ca.sqrt(rho) * v2**1.5 ])

@dataclass
class LaunchVehicle_ECI(VLType):
    def __init__(self, ScenarioDispersion: DispesrionFactorsType, LV_Configuration: int, N = [10, 60]):
        super().__init__()

        if LV_Configuration == 2: # Starship database
            self.StarShipDatabase()
        self.ApplyScenarioDispersionToLV(ScenarioDispersion)

        self.N = N
        self.scaleX = [1e7, 1e7, 1e7, 1e4, 1e4, 1e4, 1e5]
        self.scaleU = [self.SecondStage_Thrust, self.SecondStage_Thrust, self.SecondStage_Thrust]
        self.scaleT = 100.0

        sx = ca.MX.sym('sx')
        sy = ca.MX.sym('sy')
        sz = ca.MX.sym('sz')
        svx = ca.MX.sym('svx')
        svy = ca.MX.sym('svy')
        svz = ca.MX.sym('svz')
        sm = ca.MX.sym('sm')
        sfx = ca.MX.sym('sfx')
        sfy = ca.MX.sym('sfy')
        sfz = ca.MX.sym('sfz')
        sdt = ca.MX.sym('sdt')

        sym_q = ca.vertcat(sx, sy, sz, svx, svy, svz, sm)
        sym_u = ca.vertcat(sfx, sfy, sfz)

        self.nx = sym_q.size1()
        self.nu = sym_u.size1()

        x = sx * self.scaleX[0]
        y = sy * self.scaleX[1]
        z = sz * self.scaleX[2]
        vx = svx * self.scaleX[3]
        vy = svy * self.scaleX[4]
        vz = svz * self.scaleX[5]
        m = sm * self.scaleX[6]
        fx = sfx * self.scaleU[0]
        fy = sfy * self.scaleU[1]
        fz = sfz * self.scaleU[2]

        g = self.g0 * self.R0 ** 2 / (x**2 + y**2 + z**2)
        Isp = self.SecondStage_Vac_Isp * (self.SecondStage_MinThrust_IspFactor + (1.0-self.SecondStage_MinThrust_IspFactor)*(ca.norm_2(sym_u)-self.SecondStage_MinThrust_Factor)/(1.0-self.SecondStage_MinThrust_Factor))

        dx = vx
        dy = vy
        dz = vz
        dvx = fx / m - g * x / ca.sqrt(x**2 + y**2 + z**2)
        dvy = fy / m - g * y / ca.sqrt(x**2 + y**2 + z**2)
        dvz = fz / m - g * z / ca.sqrt(x**2 + y**2 + z**2)
        dm = -ca.sqrt(fx**2 + fy**2 + fz**2) / (Isp * self.g0)

        dqdt_scaled = ca.vertcat(dx / self.scaleX[0], dy / self.scaleX[1],
            dz / self.scaleX[2], dvx / self.scaleX[3],
            dvy / self.scaleX[4], dvz / self.scaleX[5],
            dm / self.scaleX[6]) * self.scaleT

        sym_sx = ca.vertcat(sx, sy, sz, svx, svy, svz, sm)
        sym_su = ca.vertcat(sfx, sfy, sfz)
        self.dynamics = ca.Function('dynamics_eci_scaled', [sym_sx, sym_su], [dqdt_scaled])
        self.dynamics_kp1 = ca.Function('dynamics_eci_kp1_scaled', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt)])
        self.ISP_calc = ca.Function('Isp_calc_scaled', [sym_sx, sym_su], [Isp])
        self.specific_acc_fun = ca.Function('Isp_calc_scaled', [sym_sx, sym_su], [ca.sqrt(fx**2+fy**2+fz**2)/m])
# You can instantiate either class directly for use in your scaled optimization pipeline.
        self.local_to_eci()
    
    def local_to_eci(self):

        local_pos_m = ca.MX.sym('local_pos_m', 3)
        local_vel_mps = ca.MX.sym('local_vel_mps', 3)
        az = ca.MX.sym('az')

        # local origin in ECI:
        lat = np.deg2rad(self.LaunchLatitude)
        lon = np.deg2rad(self.LaunchLongitude)

        # local origin altitude
        R =  self.R0 / np.sqrt(1 - self.e**2 * ca.sin(lat)**2)

        x = (R + self.LaunchAltitude) * ca.cos(lat) * ca.cos(lon)
        y = (R + self.LaunchAltitude) * ca.cos(lat) * ca.sin(lon)
        z = ((1 - self.e**2) * R + self.LaunchAltitude) * ca.sin(lat)

        eci_origin =  ca.vertcat(x, y, z)
        # local origin velocity
        v_x = -self.omega_earth * y
        v_y = self.omega_earth * x
        v_z = 0.0
        origin_vel_eci = ca.vertcat(v_x, v_y, v_z)

        # Build local to ECI rotation matrix
        # Local Up (Z)
        z = ca.vertcat(ca.cos(lat) * ca.cos(lon), ca.cos(lat) * ca.sin(lon), ca.sin(lat))

        # Local North
        n = ca.vertcat(-ca.sin(lat) * ca.cos(lon), -ca.sin(lat) * ca.sin(lon), ca.cos(lat))

        # Local East
        e = ca.vertcat(-ca.sin(lon), ca.cos(lon), 0.0)

        # X-axis: along azimuth (from North and East)
        x = ca.cos(az) * n + ca.sin(az) * e

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

    def local_to_eci_calc(self, p, v, az):

        local_pos_m = p
        local_vel_mps = v
        az = az

        # local origin in ECI:
        lat = np.deg2rad(self.LaunchLatitude)
        lon = np.deg2rad(self.LaunchLongitude)

        # local origin altitude
        R =  self.R0 / np.sqrt(1 - self.e**2 * ca.sin(lat)**2)

        x = (R + self.LaunchAltitude) * ca.cos(lat) * ca.cos(lon)
        y = (R + self.LaunchAltitude) * ca.cos(lat) * ca.sin(lon)
        z = ((1 - self.e**2) * R + self.LaunchAltitude) * ca.sin(lat)

        eci_origin =  ca.vertcat(x, y, z)
        # local origin velocity
        v_x = -self.omega_earth * y
        v_y = self.omega_earth * x
        v_z = 0.0
        origin_vel_eci = ca.vertcat(v_x, v_y, v_z)

        # Build local to ECI rotation matrix
        # Local Up (Z)
        z = ca.vertcat(ca.cos(lat) * ca.cos(lon), ca.cos(lat) * ca.sin(lon), ca.sin(lat))

        # Local North
        n = ca.vertcat(-ca.sin(lat) * ca.cos(lon), -ca.sin(lat) * ca.sin(lon), ca.cos(lat))

        # Local East
        e = ca.vertcat(-ca.sin(lon), ca.cos(lon), 0.0)

        # X-axis: along azimuth (from North and East)
        x = ca.cos(az) * n + ca.sin(az) * e

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

        return


@dataclass
class BoosterReturn_2D(VLType):
    def __init__(self, atmosphere: Atm_Type.AtmosphereType, LV_Configuration: int, ScenarioDispersion: DispesrionFactorsType, N: int = 10):
        super().__init__(atmosphere=atmosphere)

        if LV_Configuration == 2: # Starship database
            self.StarShipDatabase()
        self.ApplyScenarioDispersionToLV(ScenarioDispersion)

        self.N = N
        self.scaleX = [5e5, 1e5, 1e3, 1e3, 1e4]  # x, z, vx, vz, m
        self.scaleU = [self.FirstStage_SL_Thrust/3.0]  # Tfac - 3 engines out of 9
        self.scaleT = 100.0                     # dt

        # Scaled symbolic variables
        sx = ca.MX.sym('sx')
        sz = ca.MX.sym('sz')
        svx = ca.MX.sym('svx')
        svz = ca.MX.sym('svz')
        sm = ca.MX.sym('sm')
        sTfac = ca.MX.sym('sTfac')
        sdt = ca.MX.sym('sdt')

        sym_q = ca.vertcat(sx, sz, svx, svz, sm)
        sym_u = ca.vertcat(sTfac)
        self.nx = sym_q.size1()
        self.nu = sym_u.size1()

        # Unscale variables
        x = sx * self.scaleX[0]
        z = sz * self.scaleX[1]
        vx = svx * self.scaleX[2]
        vz = svz * self.scaleX[3]
        m = sm * self.scaleX[4]
        Thrust = sTfac * self.scaleU[0]

        Alt = ca.norm_2(ca.vertcat(x, self.R0 + z)) - self.R0
        rho = atmosphere.rho_fun(Alt)
        g = self.g0 * self.R0**2 / (x**2 + (self.R0 + z)**2)
        g_angle = ca.asin(x / (self.R0 + z))
        Drag = 0.5 * rho * (vx**2 + vz**2) * self.Sref * self.Booster_Cd0
        gama_v = ca.atan2(vz, vx)
        Fx = -(Drag + Thrust) * ca.cos(gama_v)
        Fz = -(Drag + Thrust) * ca.sin(gama_v)
        Isp = self.FirstStage_SL_Isp

        dx = vx
        dz = vz
        dvx = Fx / m - g * ca.sin(g_angle)
        dvz = Fz / m - g * ca.cos(g_angle)
        dm = -Thrust / (Isp * self.g0)

        dqdt_scaled = ca.vertcat(
            dx / self.scaleX[0],
            dz / self.scaleX[1],
            dvx / self.scaleX[2],
            dvz / self.scaleX[3],
            dm / self.scaleX[4]) * self.scaleT

        sym_sx = ca.vertcat(sx, sz, svx, svz, sm)
        sym_su = ca.vertcat(sTfac)
        self.dynamics = ca.Function('dynamics_return_scaled', [sym_sx, sym_su], [dqdt_scaled])
        self.dynamics_kp1 = ca.Function('dynamics_return_kp1', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt)])
        self.dynamics_kp1_M50 = ca.Function('dynamics_return_kp1_M50', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt, 50)])
        self.dynamics_kp1_M20 = ca.Function('dynamics_return_kp1_M20', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt, 20)])
        self.dynamics_kp1_M10 = ca.Function('dynamics_return_kp1_M10', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt, 10)])
        self.ISP_calc = ca.Function('Isp_return_scaled', [sym_sx, sym_su], [Isp])
        self.specific_acc_fun = ca.Function('Isp_calc_scaled', [sym_sx, sym_su], [ca.sqrt(Fx**2+Fz**2)/m])
        self.dynamic_pressure_fun = ca.Function('dynamic_pressure_scaled', [sym_sx], [0.5 * rho * (vx**2 + vz**2)])
        self.heat_flux_fun = ca.Function('heat_flux_scaled', [sym_sx], [self.Booster_k_empirical * ca.sqrt(rho) * (vx**2 + vz**2)**1.5 ])

@dataclass
class BoostBackBurn_2D(VLType):
    def __init__(self, ScenarioDispersion: DispesrionFactorsType, LV_Configuration: int):
        super().__init__()

        if LV_Configuration == 2: # Starship database
            self.StarShipDatabase()
        self.ApplyScenarioDispersionToLV(ScenarioDispersion)

        self.scaleX = [1e5, 1e5, 1e3, 1e3, 3e4]  # x, z, vx, vz, m
        self.scaleU = [self.FirstStage_Vac_Thrust/3.0, self.FirstStage_Vac_Thrust/3.0]                 # Tx, Tz
        self.scaleT = 30.0                      # dt

        sx = ca.MX.sym('sx')
        sz = ca.MX.sym('sz')
        svx = ca.MX.sym('svx')
        svz = ca.MX.sym('svz')
        sm = ca.MX.sym('sm')
        sTx = ca.MX.sym('sTx')
        sTz = ca.MX.sym('sTz')
        sdt = ca.MX.sym('sdt')

        sym_q = ca.vertcat(sx, sz, svx, svz, sm)
        sym_u = ca.vertcat(sTx, sTz)
        self.nx = sym_q.size1()
        self.nu = sym_u.size1()

        x = sx * self.scaleX[0]
        z = sz * self.scaleX[1]
        vx = svx * self.scaleX[2]
        vz = svz * self.scaleX[3]
        m = sm * self.scaleX[4]
        Tx = sTx * self.scaleU[0] * ScenarioDispersion.FirstStageThrust
        Tz = sTz * self.scaleU[1] * ScenarioDispersion.FirstStageThrust

        g = self.g0 * self.R0**2 / (x**2 + (self.R0 + z)**2)
        g_angle = ca.asin(x / (self.R0 + z))
        Isp = self.FirstStage_Vac_Isp * ScenarioDispersion.FirstStageIsp

        dx = vx
        dz = vz
        dvx = Tx / m - g * ca.sin(g_angle)
        dvz = Tz / m - g * ca.cos(g_angle)
        dm = -ca.sqrt(Tx**2 + Tz**2) / (Isp * self.g0)

        dqdt_scaled = ca.vertcat(
            dx / self.scaleX[0],
            dz / self.scaleX[1],
            dvx / self.scaleX[2],
            dvz / self.scaleX[3],
            dm / self.scaleX[4]) * self.scaleT

        sym_sx = ca.vertcat(sx, sz, svx, svz, sm)
        sym_su = ca.vertcat(sTx, sTz)
        self.dynamics = ca.Function('dynamics_boostback_scaled', [sym_sx, sym_su], [dqdt_scaled])
        self.dynamics_kp1 = ca.Function('dynamics_boostback_kp1', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt)])
        self.dynamics_kp1_M20 = ca.Function('dynamics_boostback_kp1_M20', [sym_sx, sym_su, sdt], [self.RK4(self.dynamics, sym_sx, sym_su, sdt, 20)])
        self.ISP_calc = ca.Function('Isp_boostback_scaled', [sym_sx, sym_su], [Isp])
        self.specific_acc_fun = ca.Function('Isp_calc_scaled', [sym_sx, sym_su], [ca.sqrt(Tx**2+Tz**2)/m])
