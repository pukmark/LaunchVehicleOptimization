"""Deterministic calibration tests with known factors and telemetry timing."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import DispersionFit as fitting
from LV_Type_scaled import DispesrionFactorsType, VLType
from OptimizationGUI import nominal_request
from TelemetryCSV import TelemetryData, aerodynamic_loads


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


def timed_simulation(duration=100., count=101):
    time = np.linspace(0, duration, count)
    state = np.zeros((5, count))
    state[1], state[2], state[4] = 10*time, 2*time, 500000
    return {'success': True, 'payload_mass': 10000., 't1_vec': time,
            'alt1_sol': state[1], 'x1': state}


def timed_telemetry():
    time = np.arange(121.)
    values = {'altitude': np.where(time <= 100, time/100, np.nan),
              'speed': np.where(time <= 100, 2*time, np.nan)}
    return TelemetryData(Path('timed.csv'), time-7,
                         {'primary': values, 'stage1': {}, 'stage2': {}, 'booster': {}}, 'test')


class DispersionFitTests(unittest.TestCase):
    def setUp(self):
        self.request = nominal_request(asdict(DispesrionFactorsType()), payload_mass=10000)
        self.calls = []
        self.options = dict(parameters=['FirstStageThrust'], bounds={'FirstStageThrust': (.9, 1.2)},
                            max_evaluations=25, altitude_scale=100, speed_scale=2, regularization=0)

    def solve(self, request, init_guess=None, ipopt_options=None):
        self.calls.append(deepcopy(request))
        return simulation(request['dispersion']['FirstStageThrust'], request['payload_mass_predefined'])

    def test_new_three_phase_csv_fits_upper_stage_and_only_nonexp_booster(self):
        t = np.arange(221)
        raw_time = t - 7
        timer = [f"T{'-' if value < 0 else '+'}{abs(value)//3600:02d}:"
                 f"{abs(value)//60%60:02d}:{abs(value)%60:02d}" for value in raw_time]
        returned = (t >= 105) & (t <= 205)
        frame = pd.DataFrame({
            't_video_sec': t + 100, 'time_str_primary': timer,
            'velocity1_kmh': np.where(t < 105, 2*t*3.6, np.nan),
            'altitude1_km': np.where(t < 105, t**2/1000, np.nan),
            'acceleration1_g': np.where(t < 105, 12./9.80665, np.nan),
            'velocity2_kmh': np.where(t >= 105, (2*t+100)*3.6, np.nan),
            'altitude2_km': np.where(t >= 105, 2*t**2/1000, np.nan),
            'velocity3_kmh': np.where(returned, 1.08*2*(205-t)*3.6, np.nan),
            'altitude3_km': np.where(returned, 1.08*(205-t)**2/1000, np.nan),
        })
        def solve(request, **kwargs):
            solution = recovery_simulation(request['dispersion']['BoosterStageCx0'])
            solution['acc1_sol'] = np.full(100, 12.)
            for number, time in ((2, np.arange(105., 156.)), (3, np.arange(155., 206.))):
                altitude = 2*time**2
                state = np.zeros((7, len(time)))
                state[0] = VLType.R0 + altitude
                state[4] = VLType.omega_earth * state[0] + 2*time+100
                state[6] = 50000.
                solution.update({f't{number}_vec': time, f'x{number}': state,
                                 f'alt{number}_sol': altitude})
            return solution
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'three_phases.csv'
            frame.to_csv(path, index=False)
            for recovery in ('EXP', 'ASDS', 'RTLS'):
                with self.subTest(recovery=recovery):
                    options = dict(self.options, parameters=['BoosterStageCx0'],
                                   bounds={'BoosterStageCx0': (.9, 1.2)}, primary_group='Phase columns')
                    result = fitting.fit_dispersion_to_telemetry(dict(self.request, recovery=recovery),
                                                                path, solver=solve, **options)
                    best = result['best']
                    expected = {f'{phase}.{channel}' for phase in ('stage1', 'stage2')
                                for channel in fitting.CHANNEL_UNITS}
                    if recovery != 'EXP':
                        expected |= {'booster.altitude', 'booster.speed'}
                        self.assertAlmostEqual(best['request']['dispersion']['BoosterStageCx0'], 1.08, places=3)
                    self.assertEqual(set(best['errors']), expected)
                    self.assertAlmostEqual(best['synchronization']['shift'], 7.)
                    self.assertEqual(best['errors']['stage1.acceleration']['calculated_samples'], 0)
                    self.assertGreater(best['errors']['stage1.acceleration']['measured_samples'], 0)
                    self.assertEqual(best['errors']['stage2.acceleration']['measured_samples'], 0)
                    self.assertGreater(best['errors']['stage2.acceleration']['calculated_samples'], 0)
                    self.assertEqual(result['settings']['primary_group'], 'Phase columns')

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
                self.assertAlmostEqual(sum(bounds['FirstStageIsp']), 2.)
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
        # This fixture intentionally has a longer telemetry phase than the
        # fixed-duration solver. Check factor recovery independently of timing.
        self.assertLess(result['best']['measurement_score'], result['baseline']['measurement_score'] * .01)
        self.assertAlmostEqual(result['best']['request']['dispersion']['FirstStageThrust'], 1.08, places=3)
        self.assertAlmostEqual(result['best']['synchronization']['shift'], 7., places=3)
        self.assertGreaterEqual(sync.call_count, len(self.calls))
        self.assertGreater(len({round(event['shift'], 4) for event in progress}), 2)
        self.assertTrue(checkpoints)
        for key in ('acceleration_scale', 'pressure_scale'):
            self.assertEqual(result['settings'][key], fitting.ERROR_SCALES[key])
            self.assertEqual(checkpoints[-1]['settings'][key], result['settings'][key])
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
            solution[f'acc{phase}_sol'] = np.array([3. if phase == 2 else 5.])
        t,channels=fitting.simulation_channels(solution,'stage2')
        np.testing.assert_allclose(t,[10,20,30])
        np.testing.assert_allclose(channels['speed'],[10,20,30])
        np.testing.assert_allclose(channels['specific_acceleration'], [3., 5., np.nan], equal_nan=True)
        np.testing.assert_allclose(channels['pressure'],
            aerodynamic_loads([1000, 2000, 3000], [10, 20, 30])['pressure'])
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
        simulated = dict(channels, specific_acceleration=np.ones(12))
        with patch('DispersionFit.simulation_channels', return_value=(time, simulated)):
            samples = fitting._comparison_samples({}, data, 'Stage 1 / booster',
                                                   {'shift': 0},
                                                   windows={'stage1': (0, 11), 'stage2': (0, 11)})
        self.assertEqual(len(samples), 8)
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
                    self.assertEqual(set(best['errors']), {f'stage1.{channel}' for channel in fitting.CHANNEL_UNITS})
                    self.assertEqual(best['request']['dispersion']['BoosterStageCx0'], 1.)
                else:
                    self.assertEqual(phases, {'stage1', 'booster'})
                    self.assertAlmostEqual(best['request']['dispersion']['BoosterStageCx0'], 1.08, places=3)
                    self.assertLess(best['measurement_score'], result['baseline']['measurement_score'] * .01)
                    self.assertEqual(best['errors']['booster.altitude']['unit'], 'm')
                    self.assertEqual(best['errors']['booster.speed']['unit'], 'm/s')
                    self.assertNotIn('booster.acceleration', best['errors'])
                    self.assertNotIn('booster.pressure', best['errors'])
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
                self.assertEqual({sample['channel'] for sample in booster}, {channel})
                observed = next(sample for sample in booster if sample['channel'] == channel)
                np.testing.assert_array_equal(observed['time'] + 7,
                                              np.delete(np.arange(107, 204), 160-107))
                if channel == 'altitude':
                    np.testing.assert_allclose(observed['values'], values[107:204][np.arange(97) != 160-107] * 1000)

    def test_recovery_without_booster_measurements_still_fits_ascent(self):
        data = recovery_telemetry(1.)
        data.groups['stage1'] = data.groups.pop('primary')
        data.groups['primary'] = {}
        result = fitting.fit_dispersion_to_telemetry(dict(self.request, recovery='ASDS'), data,
                    solver=lambda request, **kwargs: recovery_simulation(),
                    **dict(self.options, max_evaluations=3))
        self.assertEqual(set(result['best']['errors']), {f'stage1.{channel}' for channel in fitting.CHANNEL_UNITS})

    def test_calculated_channels_match_and_pressure_uses_case_density_once(self):
        solution, data = simulation(), telemetry(1.)
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        score = fitting.score_candidate(solution, data, samples)
        self.assertAlmostEqual(score['measurement_score'], 0.)
        for channel, unit in (('acceleration', 'm/s²'), ('pressure', 'kPa')):
            metric = score['errors'][f'stage1.{channel}']
            self.assertEqual(metric['unit'], unit)
            self.assertEqual(metric['measured_samples'], 0)
            self.assertEqual(metric['calculated_samples'], metric['samples'])
        nominal = fitting.simulation_channels(solution, 'stage1')[1]['pressure']
        solution['dispersion'] = {'AtmosphereDensity': 1.1}
        dispersed = fitting.simulation_channels(solution, 'stage1')[1]['pressure']
        np.testing.assert_allclose(dispersed, nominal * 1.1)
        pressure_samples = [s for s in samples if s['channel'] == 'pressure']
        self.assertGreater(fitting.score_candidate(solution, data, pressure_samples)['score'], 0)

    def test_measured_values_and_gap_fallback_use_matching_acceleration_definitions(self):
        solution, data = simulation(), telemetry(1.)
        # Proper acceleration differs deliberately from dv/dt (which is 2).
        solution['acc1_sol'] = np.full(100, 12.)
        measured = np.full(len(data.time), 12.)
        measured[30:50] = np.nan
        pressure = aerodynamic_loads(data.groups['primary']['altitude'] * 1000,
                                     data.groups['primary']['speed'])['pressure']
        pressure[40:60] = np.nan
        data.groups['primary'].update(acceleration=measured, pressure=pressure)
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        score = fitting.score_candidate(solution, data, samples)
        self.assertAlmostEqual(score['measurement_score'], 0.)
        for channel in ('acceleration', 'pressure'):
            metric = score['errors'][f'stage1.{channel}']
            self.assertGreater(metric['measured_samples'], 0)
            self.assertEqual(metric['calculated_samples'], 20)
        sample = next(s for s in samples if s['channel'] == 'acceleration')
        np.testing.assert_allclose(sample['values'][sample['measured']], 12)
        np.testing.assert_allclose(sample['values'][~sample['measured']], 2)
        solution['acc1_sol'] += 3
        changed = fitting.score_candidate(solution, data, samples)
        self.assertGreater(changed['errors']['stage1.acceleration']['rmse'], 0)
        self.assertAlmostEqual(changed['errors']['stage1.speed']['rmse'], 0.)

    def test_smaller_gui_scales_increase_each_new_channel_penalty(self):
        solution, data = simulation(), telemetry(1.)
        solution['acc1_sol'] = np.full(100, 12.)
        data.groups['primary']['acceleration'] = np.full(len(data.time), 15.)
        pressure = aerodynamic_loads(data.groups['primary']['altitude'] * 1000,
                                     data.groups['primary']['speed'])['pressure']
        data.groups['primary']['pressure'] = pressure + 10
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        for channel in ('acceleration', 'pressure'):
            selected = [sample for sample in samples if sample['channel'] == channel]
            key = f'{channel}_scale'
            low = fitting.score_candidate(solution, data, selected, **{key: 1.})
            high = fitting.score_candidate(solution, data, selected, **{key: 10.})
            self.assertGreater(low['score'], high['score'])
            self.assertEqual(low['errors'], high['errors'])
            residual = 3 if channel == 'acceleration' else 10
            self.assertAlmostEqual(low['measurement_score'], 2 * (np.hypot(1., residual) - 1.))

    def test_calculated_acceleration_matches_telemetry_intervals_on_coarse_mesh(self):
        solution, data = simulation(), telemetry(1.)
        solution['t1_vec'] = solution['t1_vec'][::10]
        solution['alt1_sol'] = solution['alt1_sol'][::10]
        solution['x1'] = solution['x1'][:, ::10]
        # Make the simulated speed piecewise linear, with a slope change at t=55.
        t = solution['t1_vec']
        solution['x1'][2] = 2*t + 2*np.maximum(t-55, 0)
        telemetry_time = data.time + 7
        data.groups['primary']['speed'] = 2*telemetry_time + 2*np.maximum(telemetry_time-55, 0)
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        selected = [s for s in samples if s['channel'] == 'acceleration']
        result = fitting.score_candidate(solution, data, selected)
        self.assertAlmostEqual(result['errors']['stage1.acceleration']['rmse'], 0.)

    def test_invalid_new_scales_fail_before_solving(self):
        for key in ('acceleration_scale', 'pressure_scale', 'phase_time_scale'):
            for value in (0., -1., np.nan, np.inf):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'Error scales'):
                    fitting.fit_dispersion_to_telemetry(self.request, telemetry(), solver=self.solve,
                                                       **dict(self.options, **{key: value}))
        self.assertEqual(self.calls, [])

    def test_booster_loads_alone_do_not_add_a_fit_phase(self):
        solution, data = recovery_simulation(), recovery_telemetry(1.)
        data.groups['stage1'] = data.groups.pop('primary')
        data.groups['primary'] = {}
        data.groups['booster'] = {'acceleration': np.full(len(data.time), 4.),
                                  'pressure': np.full(len(data.time), 5.)}
        solution['acc4_sol'] = np.full(len(solution['t4_vec']), 4.)
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster',
                                             {'shift': 7}, recovery='ASDS')
        booster = [s for s in samples if s['phase'] == 'booster']
        self.assertEqual(booster, [])
        score = fitting.score_candidate(solution, data, samples)
        self.assertEqual(set(score['errors']), {f'stage1.{channel}' for channel in fitting.CHANNEL_UNITS})

    def test_measured_and_calculated_booster_loads_do_not_affect_fit_score(self):
        for recovery in ('ASDS', 'RTLS'):
            with self.subTest(recovery=recovery):
                solution, data = recovery_simulation(), recovery_telemetry(1.)
                # Primary altitude/speed would allow calculated booster loads.
                samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster',
                                                     {'shift': 7}, recovery=recovery)
                before = fitting.score_candidate(solution, data, samples)
                self.assertEqual({s['channel'] for s in samples if s['phase'] == 'booster'},
                                 {'altitude', 'speed'})
                data.groups['booster'] = {'acceleration': np.full(len(data.time), 1e9),
                                          'pressure': np.full(len(data.time), 1e9)}
                # Ignored simulation acceleration must not be validated either.
                solution['acc4_sol'] = np.array([np.nan])
                solution['Qdyn4_sol'] = np.full(len(solution['t4_vec']), 1e12)
                changed = fitting._comparison_samples(solution, data, 'Stage 1 / booster',
                                                     {'shift': 7}, recovery=recovery)
                after = fitting.score_candidate(solution, data, changed)
                self.assertEqual(after, before)
                self.assertIn('stage1.acceleration', after['errors'])
                self.assertIn('stage1.pressure', after['errors'])

    def test_new_penalties_can_independently_drive_the_fit(self):
        for channel, parameter in (('acceleration', 'FirstStageThrust'),
                                   ('pressure', 'AtmosphereDensity')):
            with self.subTest(channel=channel):
                data = telemetry(1.)
                if channel == 'acceleration':
                    data.groups['primary']['acceleration'] = np.full(len(data.time), 12. * 1.08)
                else:
                    data.groups['primary']['pressure'] = 1.08 * aerodynamic_loads(
                        data.groups['primary']['altitude'] * 1000, data.groups['primary']['speed'])['pressure']
                def solve(request, **kwargs):
                    solution = simulation()
                    solution['dispersion'] = request['dispersion']
                    solution['acc1_sol'] = np.full(100, 12. * request['dispersion']['FirstStageThrust'])
                    return solution
                options = dict(self.options, parameters=[parameter], bounds={parameter: (.9, 1.2)},
                               acceleration_scale=.1, pressure_scale=.1)
                result = fitting.fit_dispersion_to_telemetry(self.request, data, solver=solve, **options)
                self.assertAlmostEqual(result['best']['request']['dispersion'][parameter], 1.08, places=3)
                self.assertLess(result['best']['measurement_score'], result['baseline']['measurement_score'] * .01)
                self.assertAlmostEqual(result['best']['errors']['stage1.speed']['rmse'], 0.)
                self.assertAlmostEqual(result['best']['errors']['stage1.altitude']['rmse'], 0.)

    def test_missing_telemetry_rows_do_not_create_residuals_or_extend_phase(self):
        solution, data = timed_simulation(), timed_telemetry()
        baseline_samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        baseline = fitting.score_candidate(solution, data, baseline_samples)
        self.assertAlmostEqual(baseline['score'], 0.)
        for channel in ('altitude', 'speed'):
            data.groups['primary'][channel][30:50] = np.nan
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        for sample in samples:
            self.assertFalse(((sample['time'] + 7 >= 30) & (sample['time'] + 7 < 50)).any())
        result = fitting.score_candidate(solution, data, samples)
        self.assertAlmostEqual(result['score'], 0.)
        self.assertEqual(result['phase_timing']['stage1']['telemetry_duration'], 100.)
        self.assertEqual(result['errors']['stage1.speed']['coverage'], 1.)
        # Even callers supplying sparse samples directly must not turn absent
        # telemetry outside the simulation into a coverage penalty.
        speed = next(sample for sample in samples if sample['channel'] == 'speed')
        sparse = dict(speed, time=np.r_[speed['time'], 1000], values=np.r_[speed['values'], np.nan])
        self.assertAlmostEqual(fitting.score_candidate(solution, data, [sparse])['score'], 0.)
        # An entirely empty phase must not require a simulated trajectory.
        data.groups['stage2'] = {'altitude': np.full(len(data.time), np.nan),
                                 'speed': np.full(len(data.time), np.nan)}
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        self.assertEqual({sample['phase'] for sample in samples}, {'stage1'})

    def test_phase_timing_penalizes_longer_and_shorter_candidates_with_exact_values(self):
        data = timed_telemetry()
        samples = fitting._comparison_samples(timed_simulation(), data, 'Stage 1 / booster', {'shift': 7})
        # Use a narrow value-fit window: a short candidate can still predict all
        # of its measurements, but must pay for the missing phase duration.
        speed = next(sample for sample in samples if sample['channel'] == 'speed')
        mask = speed['time'] + 7 <= 80
        selected = [dict(speed, time=speed['time'][mask], values=speed['values'][mask])]
        baseline = fitting.score_candidate(timed_simulation(), data, selected)
        self.assertAlmostEqual(baseline['score'], 0.)
        for duration in (90., 110., 120.):
            with self.subTest(duration=duration):
                result = fitting.score_candidate(timed_simulation(duration), data, selected)
                metric = result['phase_timing']['stage1']
                self.assertAlmostEqual(result['measurement_score'], 0.)
                self.assertEqual(metric['telemetry_duration'], 100.)
                self.assertEqual(metric['duration_error'], duration-100)
                self.assertGreater(result['score'], baseline['score'])
                self.assertEqual(metric['end_overrun'], max(duration-100, 0))
                if duration > 100:
                    self.assertGreater(metric['late_samples'], 0)
                    self.assertGreater(metric['late_penalty'], 0.)
                else:
                    self.assertEqual(metric['late_samples'], 0)
                    self.assertEqual(metric['late_penalty'], 0.)
        short = fitting.score_candidate(timed_simulation(90), data, selected)
        long = fitting.score_candidate(timed_simulation(110), data, selected)
        self.assertEqual(short['phase_timing']['stage1']['duration_penalty'],
                         long['phase_timing']['stage1']['duration_penalty'])
        self.assertGreater(long['score'], short['score'])

    def test_timing_uses_full_phase_including_nonaccelerating_tail_and_not_fit_margin(self):
        data, solution = timed_telemetry(), timed_simulation()
        data.groups['primary']['speed'] = np.where(data.time + 7 <= 100,
                                                 2*np.minimum(data.time + 7, 80), np.nan)
        solution['x1'][2] = 2*np.minimum(solution['t1_vec'], 80)
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7},
                                             windows={'stage1': (20, 60)})
        result = fitting.score_candidate(solution, data, samples)
        self.assertEqual(result['phase_timing']['stage1']['telemetry_duration'], 100.)
        self.assertEqual(result['timing_score'], 0.)
        self.assertTrue(all(sample['time'][-1] + 7 <= 60 for sample in samples))

    def test_baseline_already_later_than_telemetry_is_penalized(self):
        data, solution = timed_telemetry(), timed_simulation(120)
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7})
        result = fitting.score_candidate(solution, data, samples)
        self.assertEqual(result['phase_timing']['stage1']['telemetry_end'], 93.)
        self.assertEqual(result['phase_timing']['stage1']['end_overrun'], 20.)
        self.assertGreater(result['timing_score'], 0.)
        faster = fitting.score_candidate(solution, data, samples, phase_time_scale=1.)
        self.assertGreater(faster['timing_score'], result['timing_score'])

    def test_duration_mismatch_can_drive_search_when_measurements_match(self):
        request = deepcopy(self.request)
        request['dispersion']['FirstStageIsp'] = 1.1
        def solve(request, **kwargs):
            return timed_simulation(100*request['dispersion']['FirstStageIsp'])
        result = fitting.fit_dispersion_to_telemetry(
            request, timed_telemetry(), solver=solve,
            parameters=['FirstStageIsp'], bounds={'FirstStageIsp': (.9, 1.2)},
            max_evaluations=40, regularization=0)
        self.assertAlmostEqual(result['best']['request']['dispersion']['FirstStageIsp'], 1., places=3)
        self.assertLess(result['best']['timing_score'], result['baseline']['timing_score'] * .01)
        self.assertLess(result['best']['score'], result['baseline']['score'] * .01)
        report = fitting.fit_report(result)
        self.assertIn('phase_timing', report['best'])
        self.assertEqual(report['settings']['phase_time_scale'], 5.)

    def test_explicit_phase_timing_is_independent_for_ascent_upper_stage_and_return(self):
        time = np.arange(201.)
        groups = {'primary': {}}
        for phase, start, end in (('stage1', 0, 100), ('stage2', 100, 200), ('booster', 100, 200)):
            window = (time >= start) & (time <= end)
            groups[phase] = {'altitude': np.where(window, time/100, np.nan),
                             'speed': np.where(window, 2*time, np.nan)}
        data = TelemetryData(Path('phases.csv'), time-7, groups, 'test')
        solution = timed_simulation()
        for number, t in ((2, np.arange(100., 151.)), (3, np.arange(150., 201.))):
            state = np.zeros((7, len(t)))
            state[0] = VLType.R0 + 10*t
            state[4] = VLType.omega_earth*state[0] + 2*t
            solution.update({f't{number}_vec': t, f'alt{number}_sol': 10*t, f'x{number}': state})
        t = np.arange(100., 201.)
        state = np.zeros((len(t), 5))
        state[:, 1], state[:, 2] = 10*t, 2*t
        solution.update(t4_vec=t, alt4_sol=10*t, x4=state)
        samples = fitting._comparison_samples(solution, data, 'Phase columns', {'shift': 7}, recovery='ASDS')
        baseline = fitting.score_candidate(solution, data, samples, 'Phase columns')
        self.assertEqual(set(baseline['phase_timing']), {'stage1', 'stage2', 'booster'})
        self.assertEqual(baseline['timing_score'], 0.)
        for metric in baseline['phase_timing'].values():
            self.assertEqual(metric['telemetry_duration'], 100.)
        # A later upper stage must pay for overrunning its telemetry even when
        # its own duration is unchanged. The other phase targets stay fixed.
        candidate = deepcopy(solution)
        candidate['t2_vec'] += 2
        candidate['t3_vec'] += 2
        candidate['t4_vec'] = np.linspace(100, 220, len(t))
        result = fitting.score_candidate(candidate, data, samples, 'Phase columns')
        self.assertEqual(result['phase_timing']['stage1']['penalty'], 0.)
        self.assertEqual(result['phase_timing']['stage2']['duration_error'], 0.)
        self.assertEqual(result['phase_timing']['stage2']['end_overrun'], 2.)
        self.assertGreater(result['phase_timing']['stage2']['late_penalty'], 0.)
        self.assertEqual(result['phase_timing']['booster']['duration_error'], 20.)
        self.assertEqual(result['phase_timing']['booster']['end_overrun'], 20.)
        exp_samples = fitting._comparison_samples(solution, data, 'Phase columns', {'shift': 7}, recovery='EXP')
        exp = fitting.score_candidate(solution, data, exp_samples, 'Phase columns')
        self.assertNotIn('booster', exp['phase_timing'])

    def test_shared_primary_phase_bounds_are_frozen_at_baseline_separation(self):
        solution, data = recovery_simulation(), recovery_telemetry(1.)
        samples = fitting._comparison_samples(solution, data, 'Stage 1 / booster', {'shift': 7}, recovery='RTLS')
        baseline = fitting.score_candidate(solution, data, samples)
        candidate = deepcopy(solution)
        candidate['t1_vec'] = np.linspace(5, 115, 101)
        candidate['t4_vec'] = np.linspace(115, 205, 101)
        result = fitting.score_candidate(candidate, data, samples)
        for phase in ('stage1', 'booster'):
            for key in ('telemetry_start', 'telemetry_end', 'telemetry_duration'):
                self.assertEqual(result['phase_timing'][phase][key], baseline['phase_timing'][phase][key])
        self.assertEqual(result['phase_timing']['stage1']['end_overrun'], 10.)
        self.assertEqual(result['phase_timing']['booster']['duration_error'], -10.)

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
