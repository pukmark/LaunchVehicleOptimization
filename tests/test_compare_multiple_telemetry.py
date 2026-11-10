"""Mission setup, Earth-fixed plots, and direct-injection constraints."""
import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import casadi as ca
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import CompareMultipleTelemetry as comparison
from LV_Optimization_Type import LV_Optimization
from LV_Type_scaled import DispesrionFactorsType, VLType
from TelemetryCSV import TelemetryData


class ComparisonTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_mission_targets_recovery_and_burn_counts(self):
        sxm, csg, mrv = comparison.MISSIONS
        self.assertEqual([m.recovery for m in comparison.MISSIONS], ['ASDS', 'RTLS', 'EXP'])
        self.assertEqual([m.payload_kg for m in comparison.MISSIONS], [6950., 758., 4200.])
        for mission in comparison.MISSIONS:
            target = mission.target_orbit()
            self.assertEqual(target['apogee'], comparison.R0 + mission.apogee_km * 1000.)
            self.assertEqual(target['perigee'], comparison.R0 + mission.perigee_km * 1000.)
            self.assertAlmostEqual(target['i'], np.deg2rad(mission.inclination_deg))
        self.assertFalse(csg.target_orbit()['direct_injection'])
        self.assertFalse(sxm.target_orbit()['direct_injection'])
        self.assertEqual(mrv.launch_pad, 'SLC-40')

    def test_optimizer_receives_payload_site_and_injection_mode(self):
        with patch.object(comparison, 'LV_Optimization') as factory:
            solver = factory.return_value.SolveOptimiztion
            solver.return_value = {'success': True, 'payload_mass': 2205.}
            comparison.optimize_mission(comparison.MISSIONS[1])
            self.assertEqual(factory.call_args.kwargs['LV_Configuration'], 5)
            self.assertEqual(factory.call_args.kwargs['launch_site']['name'], 'Vandenberg')
            args = solver.call_args.kwargs
            self.assertEqual(args['payload_mass_predefined'], 758.)
            self.assertFalse(args['Target_Orbit']['direct_injection'])
            self.assertIsNone(args['output_dir'])
            unknown_payload_mission = replace(comparison.MISSIONS[2], payload_kg=None)
            comparison.optimize_mission(unknown_payload_mission)
            self.assertEqual(solver.call_args.kwargs['payload_mass_predefined'], -1.)
            comparison.optimize_mission(unknown_payload_mission, mrv_payload=7000.)
            self.assertEqual(solver.call_args.kwargs['payload_mass_predefined'], 7000.)
            solver.return_value = {'success': False, 'return_status': 'infeasible'}
            with self.assertRaisesRegex(RuntimeError, 'infeasible'):
                comparison.optimize_mission(comparison.MISSIONS[0])

    def test_plot_has_six_axes_ecef_upper_row_and_altitude_lower_row(self):
        first = np.zeros((5, 3))
        first[2] = [0., 100., 200.]
        upper = np.zeros((7, 3))
        upper[0] = VLType.R0 + 100000.
        upper[4] = VLType.omega_earth * upper[0] + 300.
        solution = dict(x1=first, t1_vec=[0., 1., 2.], alt1_sol=[0., 1000., 2000.],
                        x2=upper, t2_vec=[2., 3., 4.], alt2_sol=[2000., 3000., 4000.],
                        payload_mass=6950., LaunchAz=np.deg2rad(43.2), final_burn_delta_v=1523.4)
        telemetry = TelemetryData(Path('test.csv'), np.arange(3.),
                                  {'primary': {}, 'stage1': {'speed': np.array([0., np.nan, 200.]),
                                                           'altitude': np.array([0., 1., 2.])},
                                   'stage2': {}, 'booster': {}}, 'Mission time')
        cases = [comparison.Comparison(m, telemetry, solution, telemetry_shift=5.)
                 for m in comparison.MISSIONS]
        cases[2].payload_is_estimate = True
        cases[1].solution = dict(solution, direct_injection=True)
        figure, axes = comparison.plot_comparisons(cases)
        self.assertEqual(axes.shape, (2, 3))
        self.assertEqual(len(figure.axes), 6)
        np.testing.assert_allclose(axes[0, 0].lines[1].get_ydata(), 300., atol=1e-8)
        np.testing.assert_allclose(axes[1, 0].lines[0].get_ydata(), [0., 1., 2.])
        np.testing.assert_allclose(axes[0, 0].lines[-1].get_xdata(), [5., 6., 7.])
        self.assertTrue(np.isnan(axes[0, 0].lines[-1].get_ydata()[1]))
        self.assertIn('unverified', axes[0, 2].get_title())
        self.assertIsNone(axes[0, 0].get_legend())
        self.assertIn('Optimized launch azimuth: 43.20°', axes[0, 0].texts[0].get_text())
        self.assertIn('Second SES Δv: 1,523.4 m/s', axes[0, 0].texts[0].get_text())
        self.assertIn('Second SES Δv: N/A (single burn)', axes[0, 1].texts[0].get_text())
        self.assertFalse(axes[1, 0].texts)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'comparison.png'
            figure.savefig(path)
            self.assertGreater(path.stat().st_size, 1000)

    def test_real_telemetry_loads_and_failed_mission_does_not_stop_others(self):
        outcomes = [{'success': True, 'payload_mass': 6950.}, RuntimeError('infeasible'),
                    {'success': True, 'payload_mass': 7000.}]
        with patch.object(comparison, 'optimize_mission', side_effect=outcomes), contextlib.redirect_stdout(io.StringIO()):
            cases = comparison.run_comparisons()
        self.assertTrue(all(len(case.telemetry.time) > 100 for case in cases))
        self.assertEqual(cases[1].error, 'infeasible')
        self.assertIsNone(cases[1].solution)
        self.assertEqual(cases[2].solution['payload_mass'], 7000.)
        self.assertFalse(cases[2].payload_is_estimate)

    def test_direct_injection_builds_all_terminal_constraints_and_zero_reserve(self):
        # Read the initial trajectory rather than invoking IPOPT; this checks the
        # actual constraint construction and result serialization, not convergence.
        import LV_Optimization_Type as optimization
        diagnostic_type = optimization.InitialGuessConstraintDiagnostics
        original_add = diagnostic_type.add
        names = []

        def capture(instance, phase, name, *args, **kwargs):
            names.append(name)
            return original_add(instance, phase, name, *args, **kwargs)

        class InitialValues:
            def __init__(self, opti):
                self.opti = opti

            def value(self, expression):
                return self.opti.debug.value(expression, self.opti.initial())

        optimizer = LV_Optimization(DispesrionFactorsType())
        with patch.object(diagnostic_type, 'add', capture), \
             patch.object(ca.Opti, 'solve', lambda opti: InitialValues(opti)), \
             patch.object(ca.Opti, 'return_status', return_value='test'), \
             patch.object(ca.Opti, 'stats', return_value={'success': True}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = optimizer.SolveOptimiztion(
                RecoveryStrategy='EXP', payload_mass_predefined=2205.,
                Target_Orbit=dict(comparison.MISSIONS[1].target_orbit(), direct_injection=True),
                initial_guess_diagnostics=False, output_dir=None)
        self.assertEqual(result['propellant_mass_for_final_dv'], 0.)
        self.assertTrue(result['direct_injection'])
        self.assertEqual(result['final_burn_delta_v'], 0.)
        self.assertTrue({'Deployment energy target', 'Deployment angular momentum target',
                         'Deployment inclination target'} <= set(names))
        self.assertNotIn('Parking apogee target', names)
        self.assertNotIn('Parking perigee lower bound', names)


if __name__ == '__main__':
    unittest.main()
