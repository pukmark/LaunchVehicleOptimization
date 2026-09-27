"""Convert simulation velocity vectors to the telemetry's ECEF reference frame.

ECI and ECEF axes coincide at mission time zero, consistent with the model's
launch-site ECI initialization. Speeds are norms of the converted vectors.
"""
import numpy as np
from LV_Type_scaled import VLType


def local_to_ecef_velocity(velocity, azimuth, latitude=VLType.LaunchLatitude,
                           longitude=VLType.LaunchLongitude):
    """Rotate launch-local X (azimuth), Y, Z (up) velocity into ECEF."""
    lat, lon = np.deg2rad([latitude, longitude])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.])
    along = np.cos(azimuth)*north + np.sin(azimuth)*east
    basis = np.column_stack((along, np.cross(up, along), up))
    return basis @ np.asarray(velocity, dtype=float)


def eci_to_ecef_velocity(position, velocity, time):
    """v_ECEF = Rz(-omega*t) [v_ECI - omega cross r_ECI], in SI units."""
    position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    relative = velocity - np.cross([0., 0., VLType.omega_earth], position.T).T
    angle = VLType.omega_earth*np.asarray(time)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*relative[0] + s*relative[1],
                     -s*relative[0] + c*relative[1], relative[2]])
