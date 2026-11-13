"""Dual sign of a two-sided parameter buffer, and nominal second-stage dynamics."""
import unittest

import casadi as ca
import numpy as np

from LV_Optimization_Type import LV_Optimization, _ParameterBuffer
from LV_Type_scaled import DispesrionFactorsType, LaunchVehicle_ECI


class ParameterDualTests(unittest.TestCase):
    def test_rtls_lifted_factors_and_launch_limits_build_solver(self):
        model = LV_Optimization(DispesrionFactorsType(), 1)
        model.booster.N = 2
        model.eci.N = [2, 2]
        model.rocket_return.N = 2
        result = model.SolveOptimiztion(
            RecoveryStrategy='RTLS',
            Target_Orbit={'apogee': 6878137., 'perigee': 6878137., 'i': np.deg2rad(28.6)},
            payload_mass_predefined=-1., output_dir=None, initial_guess_diagnostics=False,
            lifted_parameters=['FirstStage_EmptyMass', 'FirstStageThrust',
                               *_ParameterBuffer.LAUNCH_LIMITS], parameter_equalities=True,
            ipopt_options={'max_iter': 0})
        self.assertEqual(result['return_status'], 'Maximum_Iterations_Exceeded')

    def test_nominal_isp_binding_matches_parametric_step(self):
        model = LaunchVehicle_ECI(DispesrionFactorsType(), 1, N=[2, 2])
        state = np.array([0.7, -0.2, 0.4, 0.3, -0.1, 0.2, 1.2])
        control = np.array([0.4, -0.2, 0.1])
        bound = np.array(model.dynamics_kp1(state, control, 0.15)).reshape(-1)
        free = np.array(model.dynamics_kp1_isp(
            state, control, 0.15, model.SecondStage_Vac_Isp)).reshape(-1)
        np.testing.assert_allclose(bound, free, rtol=1e-12, atol=1e-12)
        shifted = np.array(model.dynamics_kp1_isp(
            state, control, 0.15, 1.01 * model.SecondStage_Vac_Isp)).reshape(-1)
        self.assertGreater(abs(shifted[-1] - bound[-1]), 1e-8)

    def test_two_sided_duals_recover_signed_sensitivities(self):
        opti = ca.Opti()
        buffer = _ParameterBuffer(
            opti, DispesrionFactorsType(), ['SecondStageIsp', 'SecondStage_EmptyMass'], 1e-6)
        isp = buffer.rows['SecondStageIsp']['variable']
        dry = buffer.rows['SecondStage_EmptyMass']['variable']
        payload = opti.variable()
        opti.minimize(-payload)
        opti.subject_to(payload <= 10000 + 250 * (isp - 1.0) - 2 * dry)
        opti.set_initial(payload, 10000)
        opti.solver('ipopt', {'print_time': 0, 'ipopt': {
            'print_level': 0, 'tol': 1e-10, 'nlp_scaling_max_gradient': 10}})
        solution = opti.solve()
        report = buffer.evaluate(solution.value)
        self.assertAlmostEqual(report['SecondStageIsp']['dual_sensitivity'], 250, places=6)
        self.assertAlmostEqual(report['SecondStage_EmptyMass']['dual_sensitivity'], -2, places=6)
        self.assertAlmostEqual(report['SecondStage_EmptyMass']['payload_sensitivity'], -2, places=6)
        self.assertAlmostEqual(report['SecondStageIsp']['value'], 1 + 1e-6, places=6)
        self.assertAlmostEqual(report['SecondStage_EmptyMass']['value'], -1e-6, places=6)

    def test_equality_multiplier_is_the_signed_payload_derivative(self):
        opti = ca.Opti()
        buffer = _ParameterBuffer(
            opti, DispesrionFactorsType(), ['SecondStageIsp', 'SecondStage_EmptyMass'], 1e-6,
            equalities=True)
        isp = buffer.rows['SecondStageIsp']['variable']
        dry = buffer.rows['SecondStage_EmptyMass']['variable']
        payload = opti.variable()
        opti.minimize(-payload)
        opti.subject_to(payload <= 10000 + 250 * (isp - 1.0) - 2 * dry)
        opti.set_initial(payload, 10000)
        opti.solver('ipopt', {'print_time': 0, 'ipopt': {
            'print_level': 0, 'tol': 1e-12, 'nlp_scaling_max_gradient': 10}})
        solution = opti.solve()
        report = buffer.evaluate(solution.value)
        self.assertAlmostEqual(report['SecondStageIsp']['equality_dual'], 250, places=6)
        self.assertAlmostEqual(report['SecondStage_EmptyMass']['equality_dual'], -2, places=6)
        self.assertAlmostEqual(report['SecondStage_EmptyMass']['payload_sensitivity'], -2, places=6)

    def test_unknown_parameter_is_rejected(self):
        opti = ca.Opti()
        with self.assertRaises(ValueError):
            _ParameterBuffer(opti, DispesrionFactorsType(), ['NotADispersionFactor'], 1e-6)

    def test_nominal_scales_match_the_built_vehicle(self):
        from LV_Type_scaled import BoosterLaunchVehicle_2D, VLType
        import Atmosphere_Type as atmosphere_type
        dispersion = DispesrionFactorsType()
        model = VLType()
        model.ApplyVehicleConfiguration(1)
        model.ApplyScenarioDispersionToLV(dispersion)
        opti = ca.Opti()
        buffer = _ParameterBuffer(opti, dispersion, [], 1e-6)
        masses = buffer.masses(1, dispersion)
        self.assertAlmostEqual(masses['stack'], model.LV_total_mass)
        self.assertAlmostEqual(masses['second_full'], model.SecondStage_FullMass)
        self.assertAlmostEqual(masses['second_empty'], model.SecondStage_EmptyMass)
        self.assertAlmostEqual(masses['first_empty'], model.EmptyFirstStageMass)
        booster = BoosterLaunchVehicle_2D(atmosphere_type.AtmosphereType(), 1, dispersion, N=2)
        state = np.array([0.1, 0.2, 0.3, 0.4, 2.0])
        control = np.array([0.8, 0.2])
        bound = np.array(booster.dynamics_kp1(state, control, 0.1)).reshape(-1)
        free = np.array(booster.dynamics_kp1_p(state, control, 0.1, np.ones(4))).reshape(-1)
        np.testing.assert_allclose(bound, free, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
