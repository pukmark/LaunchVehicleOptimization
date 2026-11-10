"""Numerical and worker/UI checks for maximum-payload sensitivity sweeps."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from LV_Type_scaled import DispesrionFactorsType, VLType
from OptimizationGUI import ComparisonCase, OptimizationApp, nominal_request
from PayloadSensitivity import (compute_sensitivities, default_steps, draw_sensitivities,
                                validate_steps, worker, LAUNCH_PARAMETERS, parameter_values)


class PayloadSensitivityTests(unittest.TestCase):
    def request(self):
        return nominal_request(asdict(DispesrionFactorsType()), orbit_values={
            'apogee': '500', 'perigee': '300', 'inclination': '70'}, payload_mass=9000,
            launch_site={'name': 'Custom', 'latitude': 34., 'longitude': -120., 'altitude': 200.})

    def test_nonlinear_derivatives_zero_offsets_and_snapshot_isolation(self):
        request = self.request()
        original = deepcopy(request)
        calls = []

        def solve(candidate, init_guess=None):
            calls.append((deepcopy(candidate), deepcopy(init_guess)))
            factors = candidate['dispersion']
            payload = 10000 + 500 * factors['SecondStageIsp'] ** 2 - 2 * factors['SecondStage_EmptyMass']
            candidate['target_orbit']['i'] = -1  # A solver cannot mutate the saved baseline.
            return {'success': True, 'payload_mass': np.array(payload), 'tag': 'baseline'}

        checkpoints = []
        report = compute_sensitivities(request, {'SecondStageIsp': .01, 'SecondStage_EmptyMass': 40},
                                       solve=solve, checkpoint=checkpoints.append)
        self.assertEqual(request, original)
        self.assertEqual(report['evaluations'], 5)
        self.assertEqual(report['termination'], 'Complete')
        isp, mass = report['rows']
        self.assertAlmostEqual(isp['derivative'], 1000)
        self.assertAlmostEqual(isp['delta_payload'], 10)
        self.assertAlmostEqual(isp['percent'], 100 * 10 / 10500)
        self.assertAlmostEqual(mass['derivative'], -2)
        self.assertAlmostEqual(mass['delta_payload'], -80)
        for candidate, guess in calls:
            self.assertEqual(candidate['payload_mass_predefined'], -1)
            self.assertEqual(candidate['launch_site'], request['launch_site'])
            self.assertEqual(candidate['target_orbit'], request['target_orbit'])
        self.assertEqual([guess['payload_mass'] for _, guess in calls[1:]], [10500] * 4)
        self.assertEqual(checkpoints[0]['rows'], [])
        json.dumps(report, allow_nan=False)

    def test_defaults_use_physical_masses_instead_of_zero_dispersion_values(self):
        request = self.request()
        steps = default_steps(request)
        self.assertEqual(set(steps), set(asdict(DispesrionFactorsType())) | set(LAUNCH_PARAMETERS))
        self.assertAlmostEqual(steps['SecondStage_EmptyMass'], 40)
        self.assertAlmostEqual(steps['FirstStage_PropMass'], 3957)
        self.assertEqual(steps['LaunchAltDelta'], 100)
        self.assertEqual(steps['StagePartitionDelta'], .001)
        self.assertEqual(validate_steps(request, steps), steps)
        for name in LAUNCH_PARAMETERS:
            self.assertAlmostEqual(steps[name], .01 * parameter_values(request)[name])

    def test_physical_limits_hold_other_vehicle_parameters_and_dispersion_fixed(self):
        request = self.request()
        request['configuration'] = 5
        request['dispersion']['FirstStageThrust'] = .93
        request['dispersion']['FirstStage_EmptyMass'] = 100.
        original = deepcopy(request)
        coefficients = dict(zip(LAUNCH_PARAMETERS, (2., 3., -4., 5., -6.)))
        steps = {name: default_steps(request)[name] for name in LAUNCH_PARAMETERS}
        calls = []

        def solve(candidate, init_guess=None):
            calls.append(deepcopy(candidate))
            model = VLType()
            model.ApplyVehicleConfiguration(candidate['configuration'])
            model.ApplyScenarioDispersionToLV(DispesrionFactorsType(**candidate['dispersion']))
            return dict(success=True, payload_mass=1e6 + sum(
                coefficients[name] * getattr(model, name) for name in LAUNCH_PARAMETERS),
                configuration=candidate['configuration'])

        report = compute_sensitivities(request, steps, solve=solve)
        self.assertEqual(request, original)
        self.assertEqual(report['evaluations'], 11)
        baseline_parameters = calls[0]['configuration']['parameters']
        for row in report['rows']:
            self.assertAlmostEqual(row['derivative'], coefficients[row['parameter']], places=7)
        for candidate in calls:
            self.assertEqual(candidate['dispersion'], request['dispersion'])
            changed = [name for name, value in candidate['configuration']['parameters'].items()
                       if value != baseline_parameters[name]]
            self.assertLessEqual(len(changed), 1)
            self.assertTrue(set(changed) <= set(LAUNCH_PARAMETERS))
        nominal = VLType()
        nominal.ApplyVehicleConfiguration(request['configuration'])
        self.assertAlmostEqual(baseline_parameters['FirstStage_SL_Thrust'], nominal.FirstStage_SL_Thrust)

    def test_invalid_launch_limit_steps_rejected_before_solving(self):
        request = self.request()
        for name in LAUNCH_PARAMETERS:
            solve = Mock()
            with self.subTest(parameter=name), self.assertRaises(ValueError):
                compute_sensitivities(request, {name: 2 * parameter_values(request)[name]}, solve=solve)
            solve.assert_not_called()

    def test_multiple_recoveries_have_independent_baselines_guesses_and_derivatives(self):
        request = self.request()
        calls, checkpoints = [], []
        coefficients = dict(EXP=100., ASDS=200., RTLS=300.)

        def solve(candidate, init_guess=None):
            calls.append((candidate['recovery'], deepcopy(init_guess)))
            strategy = candidate['recovery']
            if init_guess is not None and init_guess.get('tag') != 'initial':
                self.assertEqual(init_guess['strategy'], strategy)
            return dict(success=True, payload_mass=10000 + coefficients[strategy] * candidate['dispersion']['FirstStageIsp'],
                        strategy=strategy)

        report = compute_sensitivities(request, {'FirstStageIsp': .01}, solve=solve,
                                       recoveries=['EXP', 'ASDS', 'RTLS'],
                                       initial_solution={'tag': 'initial'}, checkpoint=checkpoints.append)
        self.assertEqual(report['evaluations'], 9)
        self.assertEqual(report['total'], 9)
        self.assertEqual(report['termination'], 'Complete')
        self.assertEqual(len(report['runs']), 3)
        self.assertEqual([row['recovery'] for row in report['rows']], ['EXP', 'ASDS', 'RTLS'])
        for row in report['rows']:
            self.assertAlmostEqual(row['derivative'], coefficients[row['recovery']])
        for strategy, guess in calls[::3]:
            self.assertEqual(guess, {'tag': 'initial'} if strategy == request['recovery'] else None)
        self.assertEqual([r['evaluations'] for r in checkpoints], sorted(r['evaluations'] for r in checkpoints))
        json.dumps(report, allow_nan=False)
        figure = Figure()
        draw_sensitivities(figure, report)
        FigureCanvasAgg(figure).draw()
        self.assertEqual(len(figure.axes[0].patches), 3)
        self.assertTrue(all(any(strategy in label.get_text() for label in figure.axes[0].get_yticklabels())
                            for strategy in coefficients))

        app = OptimizationApp.__new__(OptimizationApp)
        app.selected_case = 0
        app.cases = [ComparisonCase('Case 1', 'blue', sensitivity_report=report,
                                    sensitivity_recoveries=['EXP', 'ASDS', 'RTLS'])]
        app.sensitivity_table = Mock(get_children=Mock(return_value=()))
        app.sensitivity_summary, app.sensitivity_canvas = Mock(), Mock()
        app.sensitivity_figure = Figure()
        app.sensitivity_form_request = Mock(return_value=request)
        app.render_sensitivity_report()
        self.assertEqual(len(app.sensitivity_table.insert.call_args_list), 3)
        self.assertEqual(app.sensitivity_table.insert.call_args_list[1].kwargs['text'], 'ASDS · FirstStageIsp')
        self.assertIn('ASDS: 10,200.00 kg', app.sensitivity_summary.set.call_args.args[0])

    def test_invalid_recovery_selections_rejected_before_solving(self):
        for strategies in ([], ['unknown'], ['ASDS', 'invalid']):
            solve = Mock()
            with self.subTest(strategies=strategies), self.assertRaises(ValueError):
                compute_sensitivities(self.request(), {'FirstStageIsp': .01},
                                      recoveries=strategies, solve=solve)
            solve.assert_not_called()

    def test_failed_recovery_baseline_does_not_stop_next_strategy(self):
        def solve(candidate, init_guess=None):
            return dict(success=candidate['recovery'] != 'ASDS', payload_mass=10000., return_status='infeasible')

        report = compute_sensitivities(self.request(), {'FirstStageIsp': .01}, solve=solve,
                                       recoveries=['ASDS', 'RTLS'])
        self.assertEqual(report['evaluations'], 4)
        self.assertEqual(len(report['rows']), 1)
        self.assertEqual(report['rows'][0]['recovery'], 'RTLS')
        self.assertIn('failed', report['termination'])

    def test_cancellation_stops_recovery_loop_and_keeps_partial_results(self):
        cancelled = False

        def checkpoint(report):
            nonlocal cancelled
            cancelled = report['evaluations'] >= 3

        solve = Mock(return_value=dict(success=True, payload_mass=10000.))
        report = compute_sensitivities(self.request(), {'FirstStageIsp': .01}, solve=solve,
                                       recoveries=['ASDS', 'RTLS'], checkpoint=checkpoint,
                                       cancel_callback=lambda: cancelled)
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(len(report['runs']), 1)
        self.assertIn('Cancelled', report['termination'])

    def test_invalid_steps_rejected_before_solving(self):
        for steps in ({}, {'unknown': 1}, {'SecondStageIsp': 0}, {'SecondStageIsp': np.nan},
                      {'SecondStageIsp': 1}, {'SecondStage_EmptyMass': 4001},
                      {'StagePartitionDelta': .2}, {'LaunchAltDelta': 7e6}, {'FirstStageIsp': 1e-30}):
            with self.subTest(steps=steps):
                solve = Mock()
                with self.assertRaises(ValueError):
                    compute_sensitivities(self.request(), steps, solve=solve)
                solve.assert_not_called()

    def test_failed_and_nonfinite_solves_never_produce_derivatives(self):
        solve = Mock(side_effect=[{'success': True, 'payload_mass': 10000},
                                  {'success': False, 'payload_mass': 20000, 'return_status': 'Infeasible'},
                                  {'success': True, 'payload_mass': np.nan}])
        report = compute_sensitivities(self.request(), {'FirstStageIsp': .01}, solve=solve)
        self.assertIsNone(report['rows'][0]['derivative'])
        self.assertEqual(report['rows'][0]['status'], 'Failed')
        self.assertEqual(set(report['rows'][0]['errors']), {'minus', 'plus'})
        json.dumps(report, allow_nan=False)

    def test_baseline_failure_stops_sweep(self):
        solve = Mock(side_effect=RuntimeError('solver unavailable'))
        report = compute_sensitivities(self.request(), {'FirstStageIsp': .01}, solve=solve)
        self.assertEqual(report['evaluations'], 1)
        self.assertEqual(report['rows'], [])
        self.assertIn('solver unavailable', report['baseline']['error'])

    def test_cancel_keeps_completed_rows_without_extra_solves(self):
        cancelled = False

        def checkpoint(report):
            nonlocal cancelled
            if report['rows'] and report['rows'][0]['derivative'] is not None:
                cancelled = True

        solve = Mock(return_value={'success': True, 'payload_mass': 10000})
        report = compute_sensitivities(self.request(), {'FirstStageIsp': .01, 'SecondStageIsp': .01},
                                       solve=solve, checkpoint=checkpoint, cancel_callback=lambda: cancelled)
        self.assertEqual(report['evaluations'], 3)
        self.assertEqual(len(report['rows']), 1)
        self.assertEqual(report['rows'][0]['derivative'], 0)
        self.assertIn('Cancelled', report['termination'])

    def test_worker_persists_json_report_and_plot_ranks_signed_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'sensitivity_inputs.json').write_text(json.dumps(dict(
                base_request=self.request(), steps={'FirstStageIsp': .01})))
            solve = Mock(side_effect=[{'success': True, 'payload_mass': 10000},
                                      {'success': True, 'payload_mass': 9900},
                                      {'success': True, 'payload_mass': 10100}])
            with patch('PayloadSensitivity._solve_candidate', solve):
                worker(path)
            report = json.loads((path / 'sensitivity_report.json').read_text())
            figure = Figure()
            draw_sensitivities(figure, report)
            self.assertAlmostEqual(figure.axes[0].patches[0].get_width(), 100)
            report['rows'].append(dict(parameter='Failed', delta_payload=None))
            draw_sensitivities(figure, report)
            self.assertEqual(len(figure.axes[0].patches), 1)

    def test_gui_launch_forces_max_payload_and_retains_case_results(self):
        app = OptimizationApp.__new__(OptimizationApp)
        app.process = None
        app.selected_case = 0
        app.cases = [ComparisonCase('Case 1', 'blue', result={'payload_mass': 9000})]
        app.cases[0].sensitivity_settings = {'SecondStageIsp': .01}
        app.cases[0].sensitivity_recoveries = ['ASDS', 'RTLS']
        app.sensitivity_form_request = Mock(return_value=self.request())
        app.capture_inputs = Mock(return_value={'test': 'draft'})
        app.log, app.notebook, app.root = Mock(), Mock(), Mock()
        app.tabs = {'Sensitivity': 'sensitivity-tab'}
        app.append_log, app.set_running, app.poll = Mock(), Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            app.output_root = Path(directory)
            with patch('OptimizationGUI.subprocess.Popen', return_value=Mock()) as popen:
                app.run_sensitivities()
            submitted = json.loads((app.run_dir / 'sensitivity_inputs.json').read_text())
            self.assertEqual(submitted['base_request']['payload_mass_predefined'], -1)
            self.assertEqual(submitted['recoveries'], ['ASDS', 'RTLS'])
            self.assertEqual(submitted['base_request']['target_orbit'], self.request()['target_orbit'])
            self.assertEqual(app.cases[0].result, {'payload_mass': 9000})
            self.assertEqual(app.job_kind, 'sensitivity')
            self.assertIn('PayloadSensitivity.py', popen.call_args.args[0][1])
            app.notebook.select.assert_called_once_with('sensitivity-tab')
            app.log_stream.close()


if __name__ == '__main__':
    unittest.main()
