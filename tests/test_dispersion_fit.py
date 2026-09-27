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


class DispersionFitTests(unittest.TestCase):
    def setUp(self):
        self.request = nominal_request(asdict(DispesrionFactorsType()), payload_mass=10000)
        self.calls = []
        self.options = dict(parameters=['FirstStageThrust'], bounds={'FirstStageThrust': (.9, 1.2)},
                            max_evaluations=25, altitude_scale=100, speed_scale=2, regularization=0)

    def solve(self, request, init_guess=None, ipopt_options=None):
        self.calls.append(deepcopy(request))
        return simulation(request['dispersion']['FirstStageThrust'], request['payload_mass_predefined'])

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
