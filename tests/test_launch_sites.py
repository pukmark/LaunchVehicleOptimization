"""Launch-site validation, model propagation, transforms, and polar edge cases."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np

from LaunchSites import LAUNCH_SITES, launch_site_inputs, parse_launch_site
from LV_Type_scaled import DispesrionFactorsType
from LV_Optimization_Type import LV_Optimization
from OptimizationGUI import nominal_request, run_optimization, local_velocity_in_ecef


class LaunchSiteTests(unittest.TestCase):
    def test_presets_custom_validation_and_request(self):
        for name in LAUNCH_SITES:
            site = parse_launch_site(launch_site_inputs(name))
            request = nominal_request(asdict(DispesrionFactorsType()), launch_site=site)
            self.assertEqual(request['launch_site'], site)
        for key, value in (('latitude', '91'), ('latitude', '-91'), ('longitude', '181'),
                           ('longitude', '-181'), ('altitude', '-6378137'),
                           ('latitude', ''), ('altitude', 'nan'), ('longitude', 'inf')):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                parse_launch_site(dict(launch_site_inputs('Custom'), **{key: value}))
        with self.assertRaises(ValueError):
            nominal_request(asdict(DispesrionFactorsType(LaunchAltDelta=-100)),
                            launch_site=dict(launch_site_inputs('Custom'), altitude=-6378100))

    def test_all_models_receive_site_before_symbolic_transforms(self):
        for name in ('Vandenberg', 'Equator', 'North Pole', 'Custom'):
            for configuration in (1, 2, 3):
                with self.subTest(site=name, configuration=configuration):
                    site = parse_launch_site(dict(launch_site_inputs(name), altitude='1000'))
                    model = LV_Optimization(DispesrionFactorsType(LaunchAltDelta=25), configuration, launch_site=site)
                    for phase in (model.booster, model.eci, model.boostback, model.rocket_return):
                        self.assertEqual(phase.LaunchLatitude, site['latitude'])
                        self.assertEqual(phase.LaunchLongitude, site['longitude'])
                        self.assertEqual(phase.LaunchAltitude, 1025)
                    position, velocity = model.eci.local_to_eci_func([0, 0, 1025], [0, 0, 0], 0)
                    position, velocity = np.asarray(position).ravel(), np.asarray(velocity).ravel()
                    lat, lon = np.deg2rad([site['latitude'], site['longitude']])
                    expected = (model.eci.R0 + 1025) * np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
                    np.testing.assert_allclose(position, expected, atol=1e-8)
                    np.testing.assert_allclose(velocity, np.cross([0, 0, model.eci.omega_earth], expected), atol=1e-10)
        nominal = LV_Optimization(DispesrionFactorsType())
        self.assertEqual(nominal.eci.LaunchLatitude, 28.6)
        self.assertEqual(nominal.eci.LaunchLongitude, -80.6)

    def test_telemetry_rotation_uses_computed_site(self):
        solution = {'LaunchAz': 0, 'launch_site': parse_launch_site(dict(launch_site_inputs('Equator'), longitude=90))}
        np.testing.assert_allclose(local_velocity_in_ecef(solution, [0, 0, 10]), [0, 10, 0], atol=1e-12)
        solution['launch_site'] = parse_launch_site(launch_site_inputs('North Pole'))
        np.testing.assert_allclose(local_velocity_in_ecef(solution, [0, 0, 10]), [0, 0, 10], atol=1e-12)

    def test_gui_boundary_passes_site_to_solver(self):
        request = nominal_request(asdict(DispesrionFactorsType()), launch_site=launch_site_inputs('Vandenberg'))
        with patch('LV_Optimization_Type.LV_Optimization') as constructor:
            run_optimization(request, None)
        self.assertEqual(constructor.call_args.kwargs['launch_site'], request['launch_site'])

    def test_equatorial_and_polar_constraints_have_finite_jacobians(self):
        test = self
        class InitialValues:
            def __init__(self, opti):
                self.opti = opti
                x = opti.debug.value(opti.x, opti.initial())
                evaluate = ca.Function('check_site', [opti.x], [opti.g, ca.jacobian(opti.g, opti.x)])
                for matrix in evaluate(x):
                    test.assertTrue(np.isfinite(np.asarray(matrix)).all())

            def value(self, expression):
                return self.opti.debug.value(expression, self.opti.initial())

        for site, inclination in (('Equator', 0), ('Equator', 180), ('North Pole', 90)):
            with self.subTest(site=site, inclination=inclination):
                request = nominal_request(asdict(DispesrionFactorsType()), launch_site=launch_site_inputs(site),
                                          orbit_values=dict(apogee=200, perigee=200, inclination=inclination))
                with patch.object(ca.Opti, 'solve', lambda opti: InitialValues(opti)), \
                     patch.object(ca.Opti, 'return_status', return_value='test'), \
                     patch.object(ca.Opti, 'stats', return_value={'success': True, 'iter_count': 0}):
                    result = run_optimization(request, None)
                self.assertEqual(result['launch_site'], request['launch_site'])
                self.assertTrue(np.isfinite(result['x2']).all())
