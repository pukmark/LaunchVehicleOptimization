#!/usr/bin/env python3

from dataclasses import dataclass, field
import numpy as np
import scipy as sp
import casadi as ca

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
class AtmosphereType(PythonMsg):

    # Altitude in meters (example: US Standard Atmosphere)
    altitudes = np.array([
        0,    1000,   5000,  10000, 15000, 20000, 25000, 30000,
    35000,  40000,  50000,  60000, 70000, 80000, 90000, 100000,
    120000, 150000, 200000, 300000, 500000, 700000, 1000000
    ])

    # Corresponding densities in kg/m³
    densities = np.array([
    1.225, 1.112, 0.7364, 0.4135, 0.1948, 0.08891, 0.04008, 0.01841,
    0.00857, 0.004, 0.00103, 0.00031, 0.00008, 0.000018, 0.000004,
    0.000001, 2e-7, 1e-8, 1e-10, 1e-14, 1e-20, 1e-26, 1e-35
    ])

    # Pressure (Pa)
    pressures = np.array([
    101325, 89875, 54048, 26436, 12040, 5475, 2549, 1197,
    552, 254, 79.7, 25.5, 8.4, 2.2, 0.55, 0.12,
    0.01, 1e-4, 1e-6, 1e-10, 1e-17, 1e-25, 1e-35
    ])

    # Speed of sound (m/s)
    speeds_of_sound = np.array([
    340.3, 336.4, 320.5, 299.5, 295.1, 295.1, 295.1, 301.5,
    308.2, 312.0, 329.8, 342.2, 350.3, 356.5, 360.6, 363.2,
    366.0, 366.7, 368.1, 368.6, 369.1, 369.3, 369.5
    ])

        # Use natural logarithm of density
    log_densities = np.log(densities)

    # interpolant on log(density)
    log_rho_interp = ca.interpolant('log_rho', 'linear', [altitudes], log_densities)
    log_p_interp = ca.interpolant('log_p', 'linear', [altitudes], np.log(pressures))

    # Linear interpolation (no log) for speed of sound
    a_interp = ca.interpolant('a', 'linear', [altitudes], speeds_of_sound)

    # Symbolic altitude
    h = ca.MX.sym("h")

    # CasADi function
    rho_fun = ca.Function("rho", [h], [ca.exp(log_rho_interp(h))])
    p_fun = ca.Function("pressure", [h], [ca.exp(log_p_interp(h))])
    a_fun = ca.Function("speed_of_sound", [h], [a_interp(h)])

    



