"""Reuse successful trajectories without accepting incompatible or broken snapshots."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from OptimizationGUI import nominal_request, run_optimization, warm_start_compatible, worker
from LV_Optimization_Type import LV_Optimization, _complete_initial_solution
from LV_Type_scaled import DispesrionFactorsType


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.request = nominal_request(asdict(DispesrionFactorsType()), recovery='ASDS')
        self.previous = deepcopy(self.request)
        self.solution = {'success': True}

    def test_nearby_dispersion_edits_reuse_successful_result(self):
        self.request['dispersion']['BoosterStageCx0'] = .9
        self.assertTrue(warm_start_compatible(self.request, self.previous, self.solution))
        self.assertFalse(warm_start_compatible(self.request, self.previous, {'success': False}))
        self.assertFalse(warm_start_compatible(self.request, None, self.solution))
        self.request['dispersion']['BoosterStageCx0'] = .5
        self.assertFalse(warm_start_compatible(self.request, self.previous, self.solution))

    def test_changed_vehicle_recovery_site_or_target_requires_a_new_seed(self):
        changes = [dict(configuration=2), dict(recovery='RTLS'), dict(payload_mass_predefined=10000),
                   dict(launch_site=dict(self.request['launch_site'], altitude=100)),
                   dict(target_orbit=dict(self.request['target_orbit'], i=.7))]
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(warm_start_compatible(dict(self.request, **change), self.previous, self.solution))

    def test_solver_and_worker_receive_private_snapshot(self):
        self.request['dispersion']['BoosterStageCx0'] = .95
        with patch('LV_Optimization_Type.LV_Optimization') as constructor:
            run_optimization(self.request, None, initial_solution=self.solution)
        self.assertIs(constructor.return_value.SolveOptimiztion.call_args.kwargs['init_guess'], self.solution)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'inputs.json').write_text(json.dumps(self.request))
            (path / 'initial_request.json').write_text(json.dumps(self.previous))
            with (path / 'initial_solution.pickle').open('wb') as stream:
                pickle.dump(self.solution, stream)
            with patch('OptimizationGUI.run_optimization', return_value={'success': True}) as solve:
                worker(path)
            self.assertEqual(solve.call_args.kwargs['initial_solution'], self.solution)
            self.previous['recovery'] = 'RTLS'
            (path / 'initial_request.json').write_text(json.dumps(self.previous))
            with patch('OptimizationGUI.run_optimization', return_value={'success': True}) as solve:
                worker(path)
            self.assertIsNone(solve.call_args.kwargs['initial_solution'])

    def test_mesh_and_finite_values_are_checked_at_the_solver_boundary(self):
        model = LV_Optimization(DispesrionFactorsType())
        solution = {name: 1. for name in ('payload_mass', 'dt1', 'dt2', 'dt3', 'LaunchAz')}
        for suffix, phase, count in (('1', model.booster, model.booster.N),
                                      ('2', model.eci, model.eci.N[0]),
                                      ('3', model.eci, model.eci.N[1])):
            solution['x' + suffix] = np.zeros((phase.nx, count + 1))
            solution['u' + suffix] = np.zeros((phase.nu, count))
        self.assertTrue(_complete_initial_solution(model, solution, 'EXP'))
        self.assertFalse(_complete_initial_solution(model, solution, 'ASDS'))
        for values in (np.zeros((model.booster.nx, 3)), np.full_like(solution['x1'], np.nan)):
            self.assertFalse(_complete_initial_solution(model, dict(solution, x1=values), 'EXP'))
        self.assertFalse(_complete_initial_solution(model, dict(solution, dt1=-1), 'EXP'))
