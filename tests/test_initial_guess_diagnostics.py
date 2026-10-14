"""Initial guesses are checked against the solver's actual phase constraints."""
import contextlib
import io
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np
import LV_Optimization_Type as optimization

from LV_Optimization_Type import InitialGuessConstraintDiagnostics, LV_Optimization
from LV_Type_scaled import DispesrionFactorsType
from LaunchSites import launch_site_inputs


class ConstraintDiagnosticTests(unittest.TestCase):
    def test_best_candidate_keeps_consistent_arrays_and_prefers_worst_violation(self):
        best = optimization._BestInitialGuess()
        states, controls = np.ones((2, 3)), np.ones((1, 2))
        best.consider([.4, 0.], (states, controls, 2.))
        states[:] = 9.
        controls[:] = 8.
        best.consider([.5, 0.], (states, controls, 3.))
        best.consider([np.nan, 0.], (states, controls, 4.))
        np.testing.assert_equal(best.candidate[0], 1.)
        np.testing.assert_equal(best.candidate[1], 1.)
        self.assertEqual(best.candidate[2], 2.)
        # A smaller worst error wins even when the sum of squares increases.
        best.consider([.3, .3], (states, controls, 5.))
        self.assertEqual(best.candidate[2], 5.)
        # With the same worst error, prefer the smaller total error.
        best.consider([.3, 0.], (states, controls, 6.))
        self.assertEqual(best.candidate[2], 6.)

    def test_scoring_uses_actual_bounds_when_reports_are_disabled(self):
        opti = ca.Opti()
        x = opti.variable(3)
        diagnostics = InitialGuessConstraintDiagnostics(opti, enabled=False)
        diagnostics.add('Phase', 'Equal', x[0] == 2.)
        diagnostics.add('Phase', 'Lower', x[1] >= 1.)
        diagnostics.add('Phase', 'Upper', x[2] <= 1.)
        opti.set_initial(x, [1., .5, .8])
        np.testing.assert_allclose(diagnostics.residuals('Phase'), [-1., .5, 0.])
        opti.set_initial(x, [2., 1., 1.])
        np.testing.assert_allclose(diagnostics.residuals('Phase'), 0.)

    def report(self, diagnostics, phase):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            report = diagnostics.report(phase)
        return report, output.getvalue()

    def test_equality_lower_upper_and_bounded_constraints_in_physical_units(self):
        opti = ca.Opti()
        x = opti.variable(4)
        diagnostics = InitialGuessConstraintDiagnostics(opti)
        diagnostics.add('Phase', 'Initial state', x[:2] == 0,
                        [100., 1000.], ['m', 'kg'], ['z', 'mass'])
        diagnostics.add('Phase', 'Pressure', x[2] <= 1., 50., 'kPa')
        diagnostics.add('Phase', 'Thrust', opti.bounded(.2, x[3], 1.))
        opti.set_initial(x, [-.01, .02, 1.4, .1])
        report, output = self.report(diagnostics, 'Phase')
        self.assertEqual(report['checked'], 4)
        self.assertEqual(len(report['violations']), 4)
        np.testing.assert_allclose([item['violation'] for item in report['violations']],
                                   [1., 20., 20., .1])
        np.testing.assert_allclose([item['scaled_residual'] for item in report['violations']],
                                   [.01, .02, .4, .1])
        self.assertIn('Initial state [mass]', output)
        self.assertIn('violation=20 kPa', output)

    def test_squared_limits_report_excess_in_original_units(self):
        opti = ca.Opti()
        alpha = opti.variable()
        diagnostics = InitialGuessConstraintDiagnostics(opti)
        diagnostics.add('Phase', 'Absolute angle', alpha**2 <= 1., 10., 'deg', squared=True)
        opti.set_initial(alpha, -1.5)
        report, _ = self.report(diagnostics, 'Phase')
        violation = report['violations'][0]
        self.assertAlmostEqual(violation['scaled_residual'], 1.25)
        self.assertAlmostEqual(violation['value'], 15.)
        self.assertAlmostEqual(violation['upper'], 10.)
        self.assertAlmostEqual(violation['violation'], 5.)

    def test_feasible_and_roundoff_residuals_are_not_reported(self):
        opti = ca.Opti()
        x = opti.variable(2)
        diagnostics = InitialGuessConstraintDiagnostics(opti)
        diagnostics.add('Phase', 'Bounds', opti.bounded(0., x, 1.))
        opti.set_initial(x, [-5e-7, 1. + 5e-7])
        report, output = self.report(diagnostics, 'Phase')
        self.assertEqual(report['violations'], [])
        self.assertIn('No constraint violations.', output)

    def test_nonfinite_constraints_are_flagged_and_other_phases_are_not_evaluated(self):
        opti = ca.Opti()
        x = opti.variable()
        y = opti.variable()
        diagnostics = InitialGuessConstraintDiagnostics(opti)
        diagnostics.add('First', 'Finite bound', x >= 0.)
        diagnostics.add('Later', 'Nonfinite division', 1. / y <= 1.)
        diagnostics.add('Later', 'Nonfinite logarithm', ca.log(y - 1.) >= 0.)
        opti.set_initial(x, 1.)
        first, _ = self.report(diagnostics, 'First')
        self.assertEqual(first['violations'], [])
        later, output = self.report(diagnostics, 'Later')
        self.assertEqual(len(later['violations']), 2)
        for violation in later['violations']:
            self.assertTrue(violation['nonfinite'])
            self.assertIsNone(violation['violation'])
        self.assertIn('NONFINITE', output)


class PhaseDiagnosticTests(unittest.TestCase):
    def solve(self, strategy, model=None, **kwargs):
        model = model or LV_Optimization(DispesrionFactorsType())
        reports = []
        captured = {}
        original_report = InitialGuessConstraintDiagnostics.report

        def report(diagnostics, phase, source='generated guess'):
            result = original_report(diagnostics, phase, source)
            if result is not None:
                reports.append(result)
            captured['diagnostics'] = diagnostics
            return result

        class InitialValues:
            def __init__(self, opti):
                self.opti = opti
                captured['opti'] = opti

            def value(self, expression):
                return self.opti.debug.value(expression, self.opti.initial())

        target = {'apogee': model.booster.R0 + 200000.,
                  'perigee': model.booster.R0 + 200000., 'i': np.deg2rad(28.6)}
        with patch.object(InitialGuessConstraintDiagnostics, 'report', report), \
             patch.object(ca.Opti, 'solve', lambda opti: InitialValues(opti)), \
             patch.object(ca.Opti, 'return_status', return_value='test'), \
             patch.object(ca.Opti, 'stats', return_value={'success': True}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            solution = model.SolveOptimiztion(RecoveryStrategy=strategy, Target_Orbit=target,
                                             output_dir=None, **kwargs)
        return solution, reports, captured, output.getvalue()

    def test_every_active_constraint_is_checked_once_after_its_phase_guess(self):
        ascent = ['First-stage ascent', 'Second stage before fairing separation',
                  'Second stage after fairing separation']
        recovery = ['Booster separation', 'Ballistic coast to reentry', 'Reentry burn',
                    'Coast before landing', 'Landing burn']
        for strategy in ('EXP', 'ASDS', 'RTLS'):
            with self.subTest(strategy=strategy):
                _, reports, captured, output = self.solve(strategy)
                expected = list(ascent)
                if strategy != 'EXP':
                    expected += list(recovery)
                if strategy == 'RTLS':
                    expected.insert(4, 'Boostback burn')
                self.assertEqual([report['phase'] for report in reports], expected)
                self.assertTrue(all('evaluation_error' not in report for report in reports))
                self.assertEqual(sum(report['checked'] for report in reports), captured['opti'].ng)
                indices = [entry[0] for entries in captured['diagnostics'].constraints.values() for entry in entries]
                self.assertEqual(sorted(indices), list(range(captured['opti'].ng)))
                for report in reports:
                    for violation in report['violations']:
                        self.assertIn('constraint', violation)
                        self.assertIn('scaled_residual', violation)
                        self.assertFalse(violation['nonfinite'])
                        self.assertFalse(violation['constraint'].startswith('Dynamics'))
                        self.assertFalse(violation['constraint'].startswith('Initial state'))
                if strategy != 'EXP':
                    burn = next(report for report in reports if report['phase'] == 'Reentry burn')
                    self.assertFalse(any(item['constraint'].startswith('Thrust factor') for item in burn['violations']))
                    ballistic = next(report for report in reports if report['phase'] == 'Ballistic coast to reentry')
                    self.assertFalse(any(item['constraint'] == 'Reentry start altitude lower bound'
                                         for item in ballistic['violations']))

    def test_supplied_warm_start_is_checked_after_overriding_generated_guess(self):
        initial, _, _, _ = self.solve('EXP')
        initial['u1'][0, 0] = 1.3
        initial['x1'][1, 0] += 1000.
        _, reports, _, output = self.solve('EXP', init_guess=initial)
        self.assertEqual(len(reports), 6)
        warm = next(report for report in reports if report['phase'] == 'First-stage ascent'
                    and report['source'] == 'supplied warm start')
        violations = {item['constraint']: item for item in warm['violations']}
        self.assertAlmostEqual(violations['Thrust factor upper bound k=0']['violation'], .3)
        self.assertAlmostEqual(violations['Initial state [z]']['violation'], 1000.)
        self.assertIn('supplied warm start', output)

    def test_reports_do_not_change_guesses_and_can_be_disabled(self):
        enabled, reports, _, _ = self.solve('ASDS', payload_mass_predefined=10000.)
        disabled, disabled_reports, _, output = self.solve('ASDS', payload_mass_predefined=10000.,
                                                          initial_guess_diagnostics=False)
        self.assertEqual(disabled_reports, [])
        self.assertNotIn('Initial guess constraints', output)
        for key in ('x1', 'x2', 'x3', 'x4_landing', 'u1', 'u2', 'u3', 'u4_landing',
                    'dt1', 'dt2', 'dt3', 'dt4_landing', 'payload_mass'):
            np.testing.assert_array_equal(enabled[key], disabled[key])
        first = reports[0]
        self.assertFalse(any(item['constraint'].startswith('Initial state') for item in first['violations']))

    def test_exhausted_phase2_search_keeps_the_last_propagated_time_step(self):
        model = LV_Optimization(DispesrionFactorsType())
        model.eci.FairingSeparationAltitude = 1e9  # Force the duration search to exhaust its budget.
        solution, reports, _, _ = self.solve('EXP', model=model)
        phase2 = next(report for report in reports if report['phase'] == 'Second stage before fairing separation')
        self.assertTrue(any(item['constraint'] == 'Fairing separation altitude' for item in phase2['violations']))
        self.assertFalse(any(item['constraint'].startswith('Dynamics') for item in phase2['violations']))
        self.assertAlmostEqual(float(solution['dt2']), (20. + 19 * 4.) / model.eci.N[0])
        self.assertTrue(np.isfinite(solution['x2']).all())

    def test_fixed_payload_and_recovery_bounds_for_all_strategies(self):
        for strategy in ('EXP', 'ASDS', 'RTLS'):
            with self.subTest(strategy=strategy):
                model = LV_Optimization(DispesrionFactorsType())
                payload = 10000.
                solution, reports, _, _ = self.solve(strategy, model=model, payload_mass_predefined=payload)
                expected_mass = (model.booster.LV_total_mass + payload - 5. * model.booster.FirstStage_SL_Thrust
                                 / (model.booster.FirstStage_SL_Isp * model.booster.g0))
                self.assertAlmostEqual(solution['x1'][4,0], expected_mass)
                self.assertEqual(solution['payload_mass'], payload)
                for report in reports:
                    self.assertFalse(any(item['constraint'].startswith(('Dynamics', 'Initial state'))
                                         or item['nonfinite'] for item in report['violations']))
                if strategy != 'EXP':
                    throttle = solution['u4_reentry'] / model.rocket_return.scaleU[0]
                    self.assertGreaterEqual(throttle, model.rocket_return.FirstStage_MinThrust_Factor)
                    self.assertLessEqual(throttle, .8)
                    thrust = np.asarray(solution['u4_landing']) / model.rocket_return.scaleU[0]
                    lower = .75/3. * model.rocket_return.EmptyFirstStageMass * model.rocket_return.g0 / model.rocket_return.scaleU[0]
                    self.assertTrue((thrust >= lower - 1e-12).all())
                    self.assertTrue((thrust <= 1./3. + 1e-12).all())
                    self.assertGreaterEqual(float(model.rocket_return.local_to_alt(solution['x4_before_reentry'])), 70000.)
                    self.assertTrue(np.isfinite(solution['x4_landing']).all())

    def test_nonfinite_partial_stage3_candidate_is_discarded_and_regenerated(self):
        model = LV_Optimization(DispesrionFactorsType())
        original_rollout = optimization._rollout_initial_guess
        dynamics = model.eci.dynamics_kp1
        injected = []

        def rollout(*args, **kwargs):
            if args[5] != 'Second stage after fairing separation' or injected:
                return original_rollout(*args, **kwargs)
            calls = []
            def fail_third_step(*values):
                calls.append(1)
                if len(calls) == 3:
                    injected.append(True)
                    return ca.DM(np.full(model.eci.nx, np.nan))
                return dynamics(*values)
            with patch.object(model.eci, 'dynamics_kp1', fail_third_step):
                return original_rollout(*args, **kwargs)

        with patch.object(optimization, '_rollout_initial_guess', rollout):
            solution, reports, _, _ = self.solve('EXP', model=model)
        self.assertEqual(injected, [True])
        self.assertTrue(np.isfinite(solution['x3']).all())
        self.assertTrue(np.isfinite(solution['u3']).all())
        phase3 = next(report for report in reports if report['phase'] == 'Second stage after fairing separation')
        self.assertFalse(any(item['constraint'].startswith(('Dynamics', 'Initial state'))
                             or item['nonfinite'] for item in phase3['violations']))

    def test_coupled_recovery_improves_seed_and_preserves_propagation(self):
        original_fit = optimization._fit_recovery_guess
        for strategy in ('ASDS', 'RTLS'):
            comparisons = []
            def fit(*args, **kwargs):
                seed, residuals = args[3], args[6]
                result = original_fit(*args, **kwargs)
                comparisons.append((optimization._guess_score(residuals(seed)),
                                    optimization._guess_score(residuals(result))))
                return result
            with self.subTest(strategy=strategy), patch.object(optimization, '_fit_recovery_guess', fit):
                model = LV_Optimization(DispesrionFactorsType())
                solution, reports, _, _ = self.solve(strategy, model=model)
                self.assertLessEqual(comparisons[0][1], comparisons[0][0])
                self.assertLess(comparisons[0][1][0], comparisons[0][0][0])
                for report in reports:
                    self.assertFalse(any(item['constraint'].startswith(('Dynamics', 'Initial state'))
                                         or item['nonfinite'] for item in report['violations']))
                if strategy == 'RTLS':
                    self.assertLess(abs(solution['x4_landing'][0,-1]), 100.)
                    self.assertLessEqual(np.linalg.norm(solution['x4_landing'][2:4,-1]), 10.)
                    self.assertGreaterEqual(solution['x4_landing'][4,-1], model.rocket_return.EmptyFirstStageMass)


class OrbitGuessTests(unittest.TestCase):
    def test_launch_heading_matches_eci_geometry_including_earth_rotation(self):
        model = LV_Optimization(DispesrionFactorsType()).eci
        state = np.array([100000., 50000., 2000., 600., 100000.])
        inclination = np.deg2rad(53.)

        def error(azimuth):
            position, velocity = model.local_to_eci_func([state[0], 0., state[1]],
                                                        [state[2], 0., state[3]], azimuth)
            h = np.cross(np.asarray(position).ravel(), np.asarray(velocity).ravel())
            return abs(h[2] / np.linalg.norm(h) - np.cos(inclination))

        heading = optimization._initial_launch_azimuth(model, state, inclination)
        self.assertLess(error(heading), 1e-8)
        self.assertLess(error(heading), error(np.pi/2. - inclination))
        model.omega_earth = 0.
        model.local_to_eci()
        unrotating_heading = optimization._initial_launch_azimuth(model, state, inclination)
        self.assertGreater(abs(heading - unrotating_heading), np.deg2rad(.5))

    def test_unreachable_direct_inclination_and_polar_sites_keep_heading_in_bounds(self):
        for site, inclination in ((None, 0.), ('Equator', 180.), ('North Pole', 90.)):
            with self.subTest(site=site):
                model = LV_Optimization(DispesrionFactorsType(),
                                        launch_site=launch_site_inputs(site) if site else None).eci
                heading = optimization._initial_launch_azimuth(
                    model, [0., model.LaunchAltitude, 2000., 0., 1.], np.deg2rad(inclination))
                self.assertTrue(np.isfinite(heading))
                self.assertGreaterEqual(heading, -np.pi/2.)
                self.assertLessEqual(heading, np.pi/2.)

    def test_final_reserve_includes_plane_change_and_leaves_required_dry_mass(self):
        model = LV_Optimization(DispesrionFactorsType()).eci
        apogee, perigee = model.R0 + 300000., model.R0 + 150000.
        semimajor = .5 * (apogee + perigee)
        v_apogee = np.sqrt(model.mu * (2./apogee - 1./semimajor))
        target = {'perigee': apogee, 'a': apogee, 'i': np.deg2rad(20.)}
        dry_mass = model.SecondStage_EmptyMass + 10000.
        state = ca.DM([apogee, 0., 0., 0., v_apogee, 0., dry_mass])
        burn = optimization._orbit_burn_quantities(model, state, target)
        v_target = np.sqrt(model.mu / apogee)
        expected_dv = np.sqrt((v_target-v_apogee)**2 +
                              2.*v_target*v_apogee*(1.-np.cos(target['i'])))
        self.assertAlmostEqual(float(burn['delta_v']), expected_dv, places=7)
        reserve = dry_mass * np.expm1(expected_dv / (.99 * model.SecondStage_Vac_Isp * model.g0))
        state[6] += reserve
        burn = optimization._orbit_burn_quantities(model, state, target)
        self.assertAlmostEqual(float(burn['mass_after_burn']), dry_mass, places=7)
        self.assertAlmostEqual(float(burn['reserve']), reserve, places=7)

    def test_final_burn_rejects_inclination_unreachable_at_apogee(self):
        model = LV_Optimization(DispesrionFactorsType()).eci
        radius = model.R0 + 300000.
        speed = np.sqrt(model.mu * (2./radius - 1./(model.R0 + 225000.)))
        state = ca.DM([0., 0., radius, speed, 0., 0., 20000.])
        target = {'perigee': radius, 'a': radius, 'i': 0.}
        burn = optimization._orbit_burn_quantities(model, state, target)
        self.assertLess(float(burn['plane_margin']), 0.)
        self.assertEqual(float(burn['mass_after_burn']), 0.)


if __name__ == '__main__':
    unittest.main()
