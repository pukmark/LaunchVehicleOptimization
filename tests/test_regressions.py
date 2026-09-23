"""Run with: MPLBACKEND=Agg python3 -m unittest discover -s tests -v."""
import contextlib
from dataclasses import fields
import io
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')

import casadi as ca
import numpy as np

import Main
from Atmosphere_Type import AtmosphereType
from LV_Optimization_Type import LV_Optimization
from LV_Type_scaled import (
    BoosterLaunchVehicle_2D, BoosterReturn_2D, BoostBackBurn_2D,
    DispesrionFactorsType, LaunchVehicle_ECI,
)


class DynamicsTests(unittest.TestCase):
    def test_thrust_and_isp_dispersion_applied_once(self):
        atmosphere = AtmosphereType()
        for factory in (
            lambda d: BoosterLaunchVehicle_2D(atmosphere, 1, d),
            lambda d: BoostBackBurn_2D(d, 1),
        ):
            nominal = factory(DispesrionFactorsType())
            varied = factory(DispesrionFactorsType(FirstStageThrust=1.1, FirstStageIsp=1.1))
            x = nominal.scale_x([0, 10000, 100, 100, 500000])
            nominal_force = nominal.dynamics(x, [1, 0]) - nominal.dynamics(x, [0, 0])
            varied_force = varied.dynamics(x, [1, 0]) - varied.dynamics(x, [0, 0])
            np.testing.assert_allclose(varied_force[2:4].full(), 1.1 * nominal_force[2:4].full())
            self.assertAlmostEqual(float(varied.ISP_calc(x, [1, 0]) / nominal.ISP_calc(x, [1, 0])), 1.1)

    def test_lift_is_perpendicular_and_drag_scales_once(self):
        atmosphere = AtmosphereType()
        nominal = BoosterLaunchVehicle_2D(atmosphere, 1, DispesrionFactorsType())
        varied = BoosterLaunchVehicle_2D(atmosphere, 1, DispesrionFactorsType(FirstStageCx0=2))
        x = nominal.scale_x([0, 1000, 100, 100, 500000])
        lift = (nominal.dynamics(x, [0, 0.5]) - nominal.dynamics(x, [0, -0.5])).full().ravel()[2:4]
        self.assertAlmostEqual(float(np.dot(lift, [100, 100])), 0)
        drag_acc = nominal.dynamics(x, [0, 0]).full().ravel()[2:4]
        varied_drag_acc = varied.dynamics(x, [0, 0]).full().ravel()[2:4]
        # Horizontal gravity is zero at x=0, so this component isolates drag.
        self.assertAlmostEqual(varied_drag_acc[0] / drag_acc[0], 2)

    def test_throttling_reduces_entire_isp(self):
        model = BoosterLaunchVehicle_2D(AtmosphereType(), 1, DispesrionFactorsType())
        for altitude in [0, 100000]:
            x = model.scale_x([0, altitude, 0, 100, 500000])
            ratio = float(model.ISP_calc(x, [model.FirstStage_MinThrust_Factor, 0]) / model.ISP_calc(x, [1, 0]))
            self.assertAlmostEqual(ratio, model.FirstStage_MinThrust_IspFactor)

    def test_return_density_affects_pressure_and_heating(self):
        atmosphere = AtmosphereType()
        nominal = BoosterReturn_2D(atmosphere, 1, DispesrionFactorsType())
        varied = BoosterReturn_2D(atmosphere, 1, DispesrionFactorsType(AtmosphereDensity=2))
        x = nominal.scale_x([0, 10000, 100, -500, 30000])
        self.assertAlmostEqual(float(varied.dynamic_pressure_fun(x) / nominal.dynamic_pressure_fun(x)), 2)
        self.assertAlmostEqual(float(varied.heat_flux_fun(x) / nominal.heat_flux_fun(x)), np.sqrt(2))

    def test_altitude_continuity_at_stage_handoff(self):
        for launch_altitude in [0, 1000, 5000]:
            model = LaunchVehicle_ECI(DispesrionFactorsType(LaunchAltDelta=launch_altitude), 1)
            for downrange, altitude in [(0, launch_altitude), (100000, launch_altitude + 80000)]:
                position, _ = model.local_to_eci_func([downrange, 0, altitude], [100, 0, 500], 0.8)
                self.assertAlmostEqual(float(ca.norm_2(position) - model.R0), model.local_to_alt([downrange, altitude]), places=7)
            position, velocity = model.local_to_eci_calc([0, 0, launch_altitude], [0, 0, 0], 0)
            np.testing.assert_allclose(velocity.full().ravel(), np.cross([0, 0, model.omega_earth], position.full().ravel()))


class SweepTests(unittest.TestCase):
    def run_sweep(self, parameter, values, solver):
        settings = dict(
            DispesrionList=[np.asarray(values)], DispesrionParamList=[parameter], ConfigList=[1],
            Target_Orbits=[{'Name': 'test'}], RecoveryStrategyVec=['EXP'],
            payload_mass_predefined=-1, Plot_interm=False, Plot_Results=0, Solution=None,
        )
        output = io.StringIO()
        with patch.dict(Main.__dict__, settings), patch.object(Main.LVopt_Type, 'LV_Optimization', solver), contextlib.redirect_stdout(output):
            Main.run_dispersion_case(0)
        return output.getvalue()

    def test_failed_case_does_not_reuse_previous_payload(self):
        class Solver:
            def __init__(self, dispersion, config):
                self.fail = dispersion.FirstStage_EmptyMass != 0

            def SolveOptimiztion(self, **kwargs):
                if self.fail:
                    raise RuntimeError('injected failure')
                return {'success': True, 'payload_mass': 12345}

        output = self.run_sweep('FirstStage_EmptyMass', [0, 100], Solver)
        self.assertIn('[12345.00, 0.00]', output)
        self.assertIn('injected failure', output)
        self.assertIn('[0.00]', self.run_sweep('FirstStage_EmptyMass', [100], Solver))

    def test_all_dispersion_fields_and_partition_alias_are_forwarded(self):
        seen = []

        class Solver:
            def __init__(self, dispersion, config):
                seen.append(dispersion)

            def SolveOptimiztion(self, **kwargs):
                return {'success': True, 'payload_mass': 1}

        for item in fields(DispesrionFactorsType):
            with self.subTest(parameter=item.name):
                self.run_sweep(item.name, [0.5, 2], Solver)
                self.assertEqual([getattr(d, item.name) for d in seen[-2:]], [0.5, 2])
        self.run_sweep('StagePartition', [-0.05, 0.05], Solver)
        self.assertEqual([d.StagePartitionDelta for d in seen[-2:]], [-0.05, 0.05])
        with self.assertRaisesRegex(ValueError, 'Unknown dispersion'):
            self.run_sweep('typo', [1], Solver)


class OptimizerOutputTests(unittest.TestCase):
    """Exercise the complete serialization path with a deterministic solver stub.

    The stub reads the actual Opti initial values, letting a returned result be
    passed through the real warm-start code without depending on convergence.
    """
    def solve(self, model, strategy, **kwargs):
        class InitialValues:
            def __init__(self, opti):
                self.opti = opti

            def value(self, expression):
                return self.opti.debug.value(expression, self.opti.initial())

        target = {'apogee': model.booster.R0 + 200000, 'perigee': model.booster.R0 + 200000, 'i': np.deg2rad(28.6)}
        kwargs.setdefault('output_dir', None)
        with patch.object(ca.Opti, 'solve', lambda opti: InitialValues(opti)), patch.object(ca.Opti, 'return_status', return_value='test'), patch.object(ca.Opti, 'stats', return_value={'success': True}):
            return model.SolveOptimiztion(RecoveryStrategy=strategy, Target_Orbit=target, **kwargs)

    def test_warm_start_round_trip_and_legacy_units(self):
        model = LV_Optimization(DispesrionFactorsType())
        original = self.solve(model, 'ASDS')
        # Explicit physical values keep this test independent of the heuristic.
        original['dt4_landing'] = np.array(1.25)
        original['u4_reentry'] = 0.8 * model.rocket_return.scaleU[0]
        restored = self.solve(model, 'ASDS', init_guess=original)
        self.assertEqual(restored['solution_units'], 'SI')
        self.assertAlmostEqual(float(restored['dt4_landing']), 1.25)
        self.assertAlmostEqual(restored['u4_reentry'], original['u4_reentry'])
        legacy = dict(original)
        del legacy['solution_units']
        legacy['dt4_landing'] = np.array(1.25 / model.rocket_return.scaleT)
        legacy['u4_reentry'] = 0.8
        restored = self.solve(model, 'ASDS', init_guess=legacy)
        self.assertAlmostEqual(float(restored['dt4_landing']), 1.25)
        self.assertAlmostEqual(restored['u4_reentry'], original['u4_reentry'])
        self.assertAlmostEqual(float(legacy['dt4_landing']), 0.0125)

    def test_fixed_payload_accepts_warm_start(self):
        model = LV_Optimization(DispesrionFactorsType())
        original = self.solve(model, 'EXP', payload_mass_predefined=10000)
        restored = self.solve(model, 'EXP', payload_mass_predefined=10000, init_guess=original)
        self.assertEqual(restored['payload_mass'], 10000)

    def test_saved_results_and_orbital_radii(self):
        with tempfile.TemporaryDirectory() as directory:
            for dispersion in [DispesrionFactorsType(), DispesrionFactorsType(FirstStage_EmptyMass=100)]:
                model = LV_Optimization(dispersion)
                result = self.solve(model, 'EXP', output_dir=directory)
                positions, velocities = result['x2'][:3], result['x2'][3:6]
                radius = np.linalg.norm(positions, axis=0)
                h = np.cross(positions.T, velocities.T).T
                eccentricity = np.linalg.norm(np.cross(velocities.T, h.T).T / model.eci.mu - positions / radius, axis=0)
                a = -model.eci.mu / (np.sum(velocities**2, axis=0) - 2 * model.eci.mu / radius)
                np.testing.assert_allclose(result['apogee2_sol'], a * (1 + eccentricity))
                np.testing.assert_allclose(result['perigee2_sol'], a * (1 - eccentricity))
            files = list(Path(directory).iterdir())
            self.assertEqual(len(files), 2)
            for path in files:
                with path.open('rb') as stream:
                    saved = pickle.load(stream)
                self.assertTrue(saved['success'])
                self.assertIn('t1_vec', saved)
                self.assertIn('dispersion', saved)
                self.assertIn('Config1', path.name)


if __name__ == '__main__':
    unittest.main()
