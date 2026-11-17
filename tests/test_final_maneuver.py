"""Smooth derivatives, orbital injection geometry and landing force controls."""
import unittest
import casadi as ca
import numpy as np

from FinalManeuver import add_final_maneuver, seed_final_maneuver
from LV_Optimization_Type import LV_Optimization, InitialGuessConstraintDiagnostics, _orbit_burn_quantities
from LV_Type_scaled import DispesrionFactorsType


class SmoothFormulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LV_Optimization(DispesrionFactorsType(), 5)

    def state(self, inclination=28.6, circular=False):
        model = self.model.eci
        radius = model.R0 + 200000.
        peri = radius if circular else model.R0 + 150000.
        speed = np.sqrt(model.mu * (2. / radius - 2. / (radius + peri)))
        angle = np.deg2rad(inclination)
        return np.array([radius, 0, 0, 0, speed*np.cos(angle), speed*np.sin(angle), 20000.])

    def target(self, inclination=30.):
        radius = self.model.eci.R0 + 200000.
        return dict(apogee=radius+100000., perigee=radius, i=np.deg2rad(inclination), a=radius+50000.)

    def test_squared_speed_is_scored_in_physical_units_during_seed_fitting(self):
        opti = ca.Opti()
        velocity = opti.variable()
        diagnostics = InitialGuessConstraintDiagnostics(opti, enabled=False)
        diagnostics.add('Landing', 'Speed', (velocity / 5.)**2 <= 1., 5., 'm/s', squared=True)
        residual = diagnostics.residual_function(['Landing'], [opti.x], lambda entry: 5.)
        self.assertAlmostEqual(float(residual([10.])), 1.)
        self.assertAlmostEqual(float(residual([-10.])), 1.)
        self.assertAlmostEqual(float(residual([4.])), 0.)

    def test_explicit_burn_matches_full_analytical_plane_change(self):
        model = self.model.eci
        for inclination in (26.48, 30., 97.9):
            with self.subTest(inclination=inclination):
                state = self.state()
                target = self.target(inclination)
                opti = ca.Opti()
                diagnostics = InitialGuessConstraintDiagnostics(opti, enabled=False)
                original_add = diagnostics.add
                def add(phase, name, *args, **kwargs):
                    if not name.startswith('Parking'):
                        original_add(phase, name, *args, **kwargs)
                diagnostics.add = add
                maneuver = add_final_maneuver(opti, diagnostics, model, ca.DM(state), target, 'Maneuver')
                opti.subject_to(maneuver['coast_angle'] == 0.)
                seed_final_maneuver(opti, maneuver, model, state, target)
                opti.minimize(maneuver['delta_v_scaled'])
                opti.solver('ipopt', {'print_time':False}, {'print_level':0, 'tol':1e-10, 'max_iter':200})
                solution = opti.solve()
                expected = _orbit_burn_quantities(model, ca.DM(state), target)
                self.assertAlmostEqual(float(solution.value(maneuver['delta_v'])), float(expected['delta_v']), places=3)
                radial = np.asarray(solution.value(maneuver['position'])).ravel()
                velocity = np.asarray(solution.value(maneuver['after'])).ravel()
                h = np.cross(radial, velocity)
                self.assertAlmostEqual(np.arccos(h[2]/np.linalg.norm(h)), target['i'], places=7)
                self.assertLess(abs(np.dot(radial, velocity))/(np.linalg.norm(radial)*np.linalg.norm(velocity)), 1e-8)

    def test_maneuver_constraint_derivatives_finite_for_circular_orbit_and_zero_delta_v(self):
        model = self.model.eci
        opti = ca.Opti()
        scaled = opti.variable(7)
        target = self.target(28.6)
        target['apogee'] = target['perigee']
        diagnostics = InitialGuessConstraintDiagnostics(opti, enabled=False)
        maneuver = add_final_maneuver(opti, diagnostics, model, model.unscale_x(scaled), target, 'Maneuver')
        state = self.state(circular=True)
        opti.set_initial(scaled, model.scale_x(state))
        seed_final_maneuver(opti, maneuver, model, state, target)
        opti.set_initial(maneuver['delta_v_scaled'], 0.)
        initial = opti.debug.value(opti.x, opti.initial())
        derivatives = ca.Function('derivatives', [opti.x], [ca.jacobian(opti.g, opti.x), ca.hessian(ca.sum1(opti.g), opti.x)[0]])
        for value in derivatives(initial):
            self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_retrograde_dynamics_and_rk4_derivatives_finite_at_zero_velocity(self):
        model = self.model.rocket_return
        state = model.scale_x([0, 10, 0, 0, model.EmptyFirstStageMass])
        x, u = ca.MX.sym('x',5), ca.MX.sym('u',1)
        rate = model.dynamics(x,u)
        step = model.dynamics_kp1(x,u,.005)
        heat_flux = model.heat_flux_fun(x)
        values = ca.Function('landing_derivatives',[x,u],[rate,ca.jacobian(rate,x),ca.hessian(ca.sum1(rate),x)[0],
                            ca.jacobian(step,x), ca.hessian(heat_flux,x)[0]])
        for value in values(state,[.2]):
            self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_return_thrust_is_regularized_retrograde_with_throttle_only(self):
        model = self.model.rocket_return
        self.assertEqual(model.dynamics.size1_in(1), 1)
        mass, throttle, epsilon = model.EmptyFirstStageMass, .2, 1e-3
        for velocity in ([0., 0.], [1e-7, -1e-7], [20., -100.]):
            with self.subTest(velocity=velocity):
                state = model.scale_x([0, 10, *velocity, mass])
                powered = np.asarray(model.dynamics(state, [throttle])).ravel()
                coast = np.asarray(model.dynamics(state, [0.])).ravel()
                acceleration = (powered - coast)[2:4] * np.asarray(model.scaleX)[2:4] / model.scaleT
                v = np.array(velocity)
                expected = -throttle * model.scaleU[0] * v / (np.sqrt(v @ v + epsilon**2) + epsilon) / mass
                np.testing.assert_allclose(acceleration, expected, atol=1e-12)
                self.assertLessEqual(np.dot(acceleration, v), 0.)
                self.assertAlmostEqual(powered[4] * model.scaleX[4] / model.scaleT,
                    -throttle * model.scaleU[0] / (model.FirstStage_SL_Isp * model.g0))


if __name__ == '__main__':
    unittest.main()
