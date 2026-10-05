"""Deterministic calibration tests with known factors and telemetry timing."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import DispersionFit as fitting
from LV_Type_scaled import DispesrionFactorsType
from OptimizationGUI import nominal_request
from TelemetryCSV import TelemetryData


def simulation(factor=1., payload=10000.):
    t = np.linspace(5, 105, 101)
    x = np.zeros((5, len(t)))
    x[1] = factor*t**2
    x[2] = factor*2*t
    x[4] = 500000
    return {'success': True, 'return_status': 'test', 'payload_mass': payload,
            't1_vec': t, 'alt1_sol': x[1], 'x1': x, 'LaunchAz': 0.}


def telemetry(factor=1.08, shift=7.):
    t = np.arange(121.)
    return TelemetryData(Path('synthetic.csv'), t-shift,
                         {'primary': {'altitude': factor*t**2/1000, 'speed': factor*2*t},
                          'stage1': {}, 'stage2': {}, 'booster': {}}, 'Synthetic mission time')


def recovery_simulation(factor=1., payload=10000.):
    result = simulation(payload=payload)
    t = np.linspace(105, 205, 101)
    x = np.zeros((len(t), 5))
    x[:, 1] = factor*(205-t)**2
    x[:, 2] = factor*2*(205-t)
    result.update(t4_vec=t.reshape(-1, 1), alt4_sol=x[:, 1].reshape(-1, 1), x4=x)
    return result


def recovery_telemetry(factor=1.08, shift=7.):
    t = np.arange(221.)
    altitude = np.where(t <= 105, t**2, factor*(205-t)**2) / 1000
    speed = np.where(t <= 105, 2*t, factor*2*(205-t))
    altitude[t > 205] = speed[t > 205] = np.nan
    return TelemetryData(Path('return.csv'), t-shift,
                         {'primary': {'altitude': altitude, 'speed': speed},
                          'stage1': {}, 'stage2': {}, 'booster': {}}, 'Synthetic mission time')


class DispersionFitTests(unittest.TestCase):
    def setUp(self):
        self.request = nominal_request(asdict(DispesrionFactorsType()), payload_mass=10000)
        self.calls = []
        self.options = dict(parameters=['FirstStageThrust'], bounds={'FirstStageThrust': (.9, 1.2)},
                            max_evaluations=25, altitude_scale=100, speed_scale=2, regularization=0)

    def solve(self, request, init_guess=None, ipopt_options=None):
        self.calls.append(deepcopy(request))
        return simulation(request['dispersion']['FirstStageThrust'], request['payload_mass_predefined'])

    def test_default_bounds_stay_nominal_while_fit_starts_from_case_values(self):
        self.request['dispersion'].update(FirstStageThrust=1.04, FirstStageIsp=.98,
                                          FirstStage_EmptyMass=500.)
        for configuration in (1, 2, 3):
            with self.subTest(configuration=configuration):
                request = dict(self.request, configuration=configuration)
                bounds = fitting.default_fit_bounds(request)
                nominal = dict(request, dispersion=asdict(DispesrionFactorsType()))
                self.assertEqual(bounds, fitting.default_fit_bounds(nominal))
                self.assertEqual(bounds['FirstStageThrust'], (.85, 1.15))
                self.assertEqual(bounds['FirstStageIsp'], (.95, 1.05))
                self.assertEqual(sum(bounds['FirstStage_EmptyMass']), 0.)
        options = {key: value for key, value in self.options.items() if key != 'bounds'}
        result = fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=self.solve, **options)
        self.assertEqual(self.calls[0]['dispersion'], self.request['dispersion'])
        self.assertEqual(result['baseline']['request']['dispersion']['FirstStageThrust'], 1.04)
        self.assertEqual(result['bounds']['FirstStageThrust'], [.85, 1.15])
        self.assertAlmostEqual(result['best']['request']['dispersion']['FirstStageThrust'], 1.08, places=3)

    def test_recovers_known_factor_and_resynchronizes_every_comparison(self):
        original = deepcopy(self.request)
        progress, checkpoints = [], []
        with patch('DispersionFit.synchronize_candidate', wraps=fitting.synchronize_candidate) as sync:
            result = fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=self.solve,
                    progress_callback=progress.append, checkpoint_callback=checkpoints.append, **self.options)
        self.assertLess(result['best']['score'], result['baseline']['score'] * .01)
        self.assertAlmostEqual(result['best']['request']['dispersion']['FirstStageThrust'], 1.08, places=3)
        self.assertAlmostEqual(result['best']['synchronization']['shift'], 7., places=3)
        self.assertGreaterEqual(sync.call_count, len(self.calls))
        self.assertGreater(len({round(event['shift'], 4) for event in progress}), 2)
        self.assertTrue(checkpoints)
        self.assertLessEqual(result['evaluations'], 25)
        self.assertEqual(self.request, original)
        self.assertTrue(all(call['payload_mass_predefined'] == 10000 for call in self.calls))
        self.assertEqual(result['best']['errors']['stage1.altitude']['unit'], 'm')
        self.assertEqual(result['best']['errors']['stage1.speed']['unit'], 'm/s')

    def test_budget_and_cancellation_return_best_checkpoint(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                self.calls.clear()
                checkpoints=[]
                options=dict(self.options, max_evaluations=4)
                result=fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=self.solve,
                            checkpoint_callback=checkpoints.append,
                            cancel_callback=lambda: cancelled and len(self.calls) >= 3, **options)
                self.assertLessEqual(len(self.calls), 4)
                self.assertIn('Cancelled' if cancelled else 'budget', result['termination'])
                self.assertEqual(result['best']['score'], checkpoints[-1]['best']['score'])

    def test_nonconverged_candidates_do_not_replace_successful_best(self):
        def solve(request, **kwargs):
            result=self.solve(request, **kwargs)
            if request['dispersion']['FirstStageThrust'] != 1.:
                result.update(success=False, return_status='infeasible test')
            return result
        result=fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=solve,
                                                   **dict(self.options, max_evaluations=5))
        self.assertEqual(result['best']['request']['dispersion']['FirstStageThrust'], 1.)
        self.assertTrue(any(item['score'] is None for item in result['history']))

    def test_payload_bootstrap_is_frozen_for_all_candidates(self):
        self.request['payload_mass_predefined'] = -1
        def solve(request, **kwargs):
            self.calls.append(deepcopy(request))
            return simulation(request['dispersion']['FirstStageThrust'], 12345)
        result=fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=solve,
                                                   **dict(self.options, max_evaluations=5))
        self.assertEqual(self.calls[0]['payload_mass_predefined'], -1)
        self.assertTrue(all(call['payload_mass_predefined'] == 12345 for call in self.calls[1:]))
        self.assertEqual(result['best']['request']['payload_mass_predefined'], 12345)
        self.assertEqual(result['history'][0]['kind'], 'payload')

    def test_current_result_can_supply_fixed_payload_without_bootstrap(self):
        self.request['payload_mass_predefined'] = -1
        result=fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=self.solve,
                      initial_solution=simulation(payload=12500), **dict(self.options, max_evaluations=3))
        self.assertEqual(result['baseline']['request']['payload_mass_predefined'], 12500)
        self.assertTrue(all(call['payload_mass_predefined'] == 12500 for call in self.calls))

    def test_invalid_bounds_and_missing_sync_data_fail_before_solving(self):
        for bounds in ({'FirstStageThrust': (1.1, 1.2)}, {'FirstStageThrust': (0, 1.2)},
                       {'FirstStageThrust': (1.1, .9)}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=self.solve,
                                                   **dict(self.options, bounds=bounds))
        data=telemetry()
        del data.groups['primary']['speed']
        with self.assertRaisesRegex(ValueError, 'speed telemetry'):
            fitting.fit_dispersion_to_telemetry(self.request, data, solver=self.solve, **self.options)
        self.assertEqual(self.calls, [])

    def test_failed_baseline_sync_and_missing_coverage_are_not_silent(self):
        data=telemetry()
        data.groups['primary']['speed'] += 1000
        with self.assertRaisesRegex(ValueError, 'baseline'):
            fitting.fit_dispersion_to_telemetry(self.request, data, solver=self.solve, **self.options)
        data=telemetry(1.)
        base=simulation()
        sync=fitting.synchronize_candidate(base,data)
        samples=fitting._comparison_samples(base,data,'Stage 1 / booster',sync)
        shortened=deepcopy(base)
        shortened['t1_vec']=base['t1_vec'][:20]
        shortened['alt1_sol']=base['alt1_sol'][:20]
        shortened['x1']=base['x1'][:,:20]
        with self.assertRaisesRegex(ValueError, '90%'):
            fitting.score_candidate(shortened,data,samples)

    def test_stage_two_comparison_uses_ecef_and_can_sync_without_stage_one(self):
        from LV_Type_scaled import VLType
        solution={}
        for phase,t in ((2,np.array([10.,20.])),(3,np.array([20.,30.]))):
            x=np.zeros((7,2)); x[0]=VLType.R0
            x[4]=VLType.omega_earth*VLType.R0 + t
            solution.update({f't{phase}_vec':t,f'x{phase}':x,f'alt{phase}_sol':t*100})
        t,channels=fitting.simulation_channels(solution,'stage2')
        np.testing.assert_allclose(t,[10,20,30])
        np.testing.assert_allclose(channels['speed'],[10,20,30])
        data=TelemetryData(Path('stage2.csv'),np.array([0.,10.,20.,30.]),
                           {'primary':{'speed':np.array([0.,10.,20.,30.])},'stage1':{},'stage2':{},'booster':{}},'test')
        synced=fitting.synchronize_candidate(solution,data,'Stage 2')
        self.assertEqual(synced['phase'],'stage2')
        self.assertAlmostEqual(synced['shift'],0.)

    def test_samples_include_only_positive_acceleration_ascent(self):
        time = np.arange(12.)
        speed = np.array([0., 2., 4., 6., 6., 5., 4., 4., 6., 8., 10., 12.])
        channels = {'speed': speed, 'altitude': time, 'acceleration': np.ones(12)}
        data = TelemetryData(Path('burns.csv'), time,
                             {'primary': {}, 'stage1': channels, 'stage2': channels,
                              'booster': channels}, 'test')
        with patch('DispersionFit.simulation_channels', return_value=(time, channels)):
            samples = fitting._comparison_samples({}, data, 'Stage 1 / booster',
                                                   {'shift': 0},
                                                   windows={'stage1': (0, 11), 'stage2': (0, 11)})
        self.assertEqual(len(samples), 4)
        for sample in samples:
            self.assertIn(sample['phase'], ('stage1', 'stage2'))
            np.testing.assert_array_equal(sample['time'], [1, 2, 3, 8, 9, 10, 11])

    def test_nonaccelerating_telemetry_cannot_produce_fit_samples(self):
        data = telemetry()
        data.groups['primary']['speed'][:] = 10.
        with self.assertRaisesRegex(ValueError, 'positive-acceleration'):
            fitting._comparison_samples(simulation(), data, 'Stage 1 / booster', {'shift': 7})

    def test_return_telemetry_drives_fit_for_asds_and_rtls_but_not_exp(self):
        options = dict(self.options, parameters=['BoosterStageCx0'],
                       bounds={'BoosterStageCx0': (.9, 1.2)})
        def solve(request, **kwargs):
            return recovery_simulation(request['dispersion']['BoosterStageCx0'],
                                       request['payload_mass_predefined'])
        for recovery in ('ASDS', 'RTLS', 'EXP'):
            with self.subTest(recovery=recovery):
                request = dict(self.request, recovery=recovery)
                result = fitting.fit_dispersion_to_telemetry(request, recovery_telemetry(),
                                                            solver=solve, **options)
                best = result['best']
                self.assertAlmostEqual(best['synchronization']['shift'], 7., places=3)
                phases = {window['phase'] for window in result['windows']}
                if recovery == 'EXP':
                    self.assertEqual(phases, {'stage1'})
                    self.assertEqual(set(best['errors']), {'stage1.altitude', 'stage1.speed'})
                    self.assertEqual(best['request']['dispersion']['BoosterStageCx0'], 1.)
                else:
                    self.assertEqual(phases, {'stage1', 'booster'})
                    self.assertAlmostEqual(best['request']['dispersion']['BoosterStageCx0'], 1.08, places=3)
                    self.assertLess(best['score'], result['baseline']['score'] * .01)
                    self.assertEqual(best['errors']['booster.altitude']['unit'], 'm')
                    self.assertEqual(best['errors']['booster.speed']['unit'], 'm/s')
                    report = fitting.fit_report(result)
                    self.assertIn('booster.speed', report['best']['errors'])

    def test_return_samples_include_coasts_and_deceleration_and_allow_single_channel(self):
        for channel in ('altitude', 'speed'):
            with self.subTest(channel=channel):
                data = recovery_telemetry(1.)
                data.groups['stage1'] = data.groups.pop('primary')
                data.groups['primary'] = {}
                t = data.time + 7
                values = (205-t)**2/1000 if channel == 'altitude' else 2*(205-t)
                # Include zero speed change within the return window.
                values[150:155] = values[150]
                values[160] = np.nan
                data.groups['booster'] = {channel: values}
                samples = fitting._comparison_samples(recovery_simulation(), data, 'Stage 1 / booster',
                                                       {'shift': 7}, recovery='ASDS')
                booster = [sample for sample in samples if sample['phase'] == 'booster']
                self.assertEqual(len(booster), 1)
                self.assertEqual(booster[0]['channel'], channel)
                np.testing.assert_array_equal(booster[0]['time'] + 7,
                                              np.delete(np.arange(107, 204), 160-107))
                if channel == 'altitude':
                    np.testing.assert_allclose(booster[0]['values'], values[107:204][np.arange(97) != 160-107] * 1000)

    def test_recovery_without_booster_measurements_still_fits_ascent(self):
        data = recovery_telemetry(1.)
        data.groups['stage1'] = data.groups.pop('primary')
        data.groups['primary'] = {}
        result = fitting.fit_dispersion_to_telemetry(dict(self.request, recovery='ASDS'), data,
                    solver=lambda request, **kwargs: recovery_simulation(),
                    **dict(self.options, max_evaluations=3))
        self.assertEqual(set(result['best']['errors']), {'stage1.altitude', 'stage1.speed'})

    def test_booster_windows_cannot_include_preseparation_samples(self):
        samples = fitting._comparison_samples(recovery_simulation(), recovery_telemetry(1.),
                                             'Stage 1 / booster', {'shift': 7},
                                             windows={'booster': (0, 300)}, recovery='RTLS')
        for sample in samples:
            if sample['phase'] == 'booster':
                self.assertGreaterEqual(sample['time'][0] + 7, 105)
                self.assertLessEqual(sample['time'][-1] + 7, 205)

    def test_return_candidates_must_cover_fixed_booster_window(self):
        data = recovery_telemetry(1.)
        solution = recovery_simulation()
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster',
                                             {'shift': 7}, recovery='ASDS')
        shortened = deepcopy(solution)
        for key in ('t4_vec', 'alt4_sol', 'x4'):
            shortened[key] = solution[key][:20]
        with self.assertRaisesRegex(ValueError, 'booster candidate covers less than 90%'):
            fitting.score_candidate(shortened, data, samples)
        for key in ('t4_vec', 'alt4_sol', 'x4'):
            with self.subTest(missing=key):
                incomplete = dict(solution)
                del incomplete[key]
                with self.assertRaisesRegex(ValueError, 'Incomplete booster simulation'):
                    fitting.score_candidate(incomplete, data, samples)

    def test_incompatible_recovery_warm_start_is_discarded(self):
        self.request['recovery'] = 'ASDS'
        previous = dict(simulation(), configuration=1)
        with patch('LV_Optimization_Type.LV_Optimization') as constructor:
            fitting._solve_candidate(self.request, init_guess=previous)
        self.assertIsNone(constructor.return_value.SolveOptimiztion.call_args.kwargs['init_guess'])
        self.request['recovery'] = 'EXP'
        with patch('LV_Optimization_Type.LV_Optimization') as constructor:
            fitting._solve_candidate(self.request, init_guess=previous)
        self.assertIs(constructor.return_value.SolveOptimiztion.call_args.kwargs['init_guess'], previous)
