"""Known-vector checks for simulation-to-telemetry velocity frame conversions."""
import unittest
import numpy as np

from TelemetryFrames import local_to_ecef_velocity, eci_to_ecef_velocity
from LV_Type_scaled import VLType


class VelocityFrameTests(unittest.TestCase):
    def test_local_axes_and_speed_invariance(self):
        local = np.array([[10., 20.], [20., 30.], [30., 40.]])
        north_launch = local_to_ecef_velocity(local, 0, latitude=0, longitude=0)
        np.testing.assert_allclose(north_launch, [local[2], -local[1], local[0]])
        east_launch = local_to_ecef_velocity(local, np.pi/2, latitude=0, longitude=0)
        np.testing.assert_allclose(east_launch, [local[2], local[0], local[1]], atol=1e-12)
        arbitrary = local_to_ecef_velocity(local, .84)
        np.testing.assert_allclose(np.linalg.norm(arbitrary, axis=0), np.linalg.norm(local, axis=0))

    def test_corotating_point_is_stationary_in_ecef(self):
        position = np.array([[7000000., 0], [0, 8000000.], [100, 200]])
        velocity = np.cross([0, 0, VLType.omega_earth], position.T).T
        np.testing.assert_allclose(eci_to_ecef_velocity(position, velocity, [0, 2000]), 0, atol=1e-12)

    def test_earth_rotation_at_vehicle_radius_and_rotating_axes(self):
        radius = 7000000.
        omega = VLType.omega_earth
        position = np.array([[radius, 0.], [0., radius], [0., 0.]])
        velocity = np.array([[0., -(100 + omega*radius)], [100 + omega*radius, 0.], [0., 0.]])
        converted = eci_to_ecef_velocity(position, velocity, [0., np.pi/(2*omega)])
        np.testing.assert_allclose(converted, [[0, 0], [100, 100], [0, 0]], atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(converted, axis=0), 100)
