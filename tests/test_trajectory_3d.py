"""3D rendering and orbit geometry checks without an interactive display."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from LV_Type_scaled import VLType, LaunchVehicle_ECI, DispesrionFactorsType
from LaunchSites import launch_site_inputs, parse_launch_site
from OptimizationGUI import ComparisonCase, OptimizationApp, RESULT_TABS
from Trajectory3D import (draw_trajectory_3d, orbit_curve, solution_orbits,
                          first_stage_eci_positions, case_results_text)


def elliptic_case():
    perigee, apogee = VLType.R0 + 150000, VLType.R0 + 500000
    semimajor = (perigee + apogee) / 2
    inclination = np.deg2rad(53)
    position = np.array([perigee, 0., 0.])
    velocity = np.sqrt(VLType.mu*(2/perigee - 1/semimajor)) * np.array([0., np.cos(inclination), np.sin(inclination)])
    state = np.zeros((7, 2))
    state[:3] = position[:, None]
    state[3:6] = velocity[:, None]
    target = {'apogee': apogee, 'perigee': VLType.R0 + 200000, 'i': inclination}
    desired_speed = np.sqrt(VLType.mu*(2/apogee - 2/(target['apogee'] + target['perigee'])))
    return {'x2': state.copy(), 'x3': state.copy(), 'v3_desired': desired_speed}, target


class Trajectory3DTests(unittest.TestCase):
    def test_ellipse_apsides_closure_and_inclination(self):
        solution, _ = elliptic_case()
        state = solution['x3'][:, -1]
        orbit = orbit_curve(state[:3], state[3:6])
        radius = np.linalg.norm(orbit['points'], axis=1)
        np.testing.assert_allclose([radius.min(), radius.max()], [VLType.R0 + 150000, VLType.R0 + 500000], rtol=1e-12)
        np.testing.assert_allclose(orbit['points'][0], orbit['points'][-1], atol=1e-8)
        np.testing.assert_allclose(orbit['points'] @ orbit['normal'], 0, atol=1e-8)
        self.assertAlmostEqual(orbit['apogee'], VLType.R0 + 500000)
        self.assertAlmostEqual(orbit['perigee'], VLType.R0 + 150000)

    def test_circular_orbit_is_finite(self):
        radius = VLType.R0 + 200000
        orbit = orbit_curve([radius, 0, 0], [0, np.sqrt(VLType.mu/radius), 0])
        self.assertTrue(np.isfinite(orbit['points']).all())
        np.testing.assert_allclose(np.linalg.norm(orbit['points'], axis=1), radius)

    def test_final_burn_at_parking_apogee_matches_saved_speed_and_target(self):
        solution, target = elliptic_case()
        original = deepcopy(solution)
        for saved_speed in (True, False):
            with self.subTest(saved_speed=saved_speed):
                candidate = dict(solution)
                if not saved_speed:
                    del candidate['v3_desired']
                parking, final = solution_orbits(candidate, target)
                self.assertAlmostEqual(parking['perigee'], VLType.R0 + 150000)
                self.assertAlmostEqual(final['perigee'], target['perigee'])
                self.assertAlmostEqual(final['apogee'], target['apogee'])
                np.testing.assert_allclose(parking['normal'], final['normal'], atol=1e-12)
        for key in ('x2', 'x3'):
            np.testing.assert_array_equal(solution[key], original[key])

    def test_unbound_or_degenerate_states_are_rejected_without_propagation(self):
        radius = VLType.R0 + 200000
        for position, velocity in (([radius, 0, 0], [0, np.sqrt(2*VLType.mu/radius)*1.01, 0]),
                                   ([radius, 0, 0], [10, 0, 0]), ([0, 0, 0], [0, 1, 0]),
                                   ([np.nan, 0, 0], [0, 1, 0])):
            with self.subTest(position=position, velocity=velocity), self.assertRaises(ValueError):
                orbit_curve(position, velocity)

    def test_render_compares_cases_in_km_with_equal_scale_and_preserves_camera(self):
        solution, target = elliptic_case()
        figure = Figure(figsize=(12, 8), constrained_layout=True)
        cases = [(solution, target, 'blue', 'Case 1'), (solution, target, 'red', 'Case 2')]
        draw_trajectory_3d(figure, cases)
        FigureCanvasAgg(figure).draw()
        self.assertEqual(len(figure.axes), 6)
        axis = figure.axes[0]
        self.assertEqual(axis.name, '3d')
        self.assertEqual(len(axis.lines), 6)
        np.testing.assert_allclose(axis.lines[0].get_data_3d()[0], np.tile(solution['x2'][0] / 1000, 2))
        self.assertEqual(axis.lines[3].get_color(), 'red')
        self.assertIn('Case 2', axis.lines[3].get_label())
        self.assertEqual(axis.get_xlim(), axis.get_ylim())
        self.assertEqual(axis.get_ylim(), axis.get_zlim())
        for projection in figure.axes[1:4]:
            self.assertEqual(len(projection.lines), 7)  # Earth + three paths per case.
        axis.view_init(elev=35, azim=20)
        draw_trajectory_3d(figure, cases[:1])
        self.assertEqual((figure.axes[0].elev, figure.axes[0].azim), (35, 20))
        self.assertEqual(len(figure.axes[0].lines), 3)

    def test_first_stage_positions_match_solver_transform_for_each_site(self):
        for name, azimuth in (('Vandenberg', -.5), ('Equator', np.pi/2), ('North Pole', 0.)):
            with self.subTest(site=name):
                site = parse_launch_site(dict(launch_site_inputs(name), altitude='1000'))
                dispersion = DispesrionFactorsType(LaunchAltDelta=25)
                model = LaunchVehicle_ECI(dispersion, 1, launch_site=site)
                state = np.array([[0., 50000.], [1025., 100000.], [100., 2000.],
                                  [50., 1000.], [500000., 200000.]])
                solution = {'x1': state, 'LaunchAz': azimuth, 'launch_site': site,
                            'configuration': 1, 'dispersion': {'LaunchAltDelta': 25}}
                points = first_stage_eci_positions(solution)
                for index in range(state.shape[1]):
                    expected, _ = model.local_to_eci_calc([state[0, index], 0, state[1, index]],
                                                        [state[2, index], 0, state[3, index]], azimuth)
                    np.testing.assert_allclose(points[index], np.asarray(expected).ravel(), atol=1e-8)
                self.assertAlmostEqual(np.linalg.norm(points[0]) - model.R0, 1025)

    def test_first_and_second_stages_render_with_separate_case_colored_results(self):
        solution, target = elliptic_case()
        solution.update(x1=np.array([[0., 50000.], [0., 150000.], [100., 2000.],
                                     [50., 1000.], [500000., 200000.]]),
                        LaunchAz=.5, payload_mass=12500., t1_vec=np.array([5., 170.]),
                        t3_vec=np.array([200., 520.]), Qdyn1_sol=np.array([5000., 29500.]),
                        propellant_mass_for_final_dv=1200.)
        original = deepcopy(solution)
        figure = Figure(figsize=(14, 9), constrained_layout=True)
        second = dict(solution, payload_mass=15000., LaunchAz=-.2)
        cases = [(solution, target, 'blue', 'Case 1'), (second, target, 'red', 'Case 2')]
        draw_trajectory_3d(figure, cases)
        FigureCanvasAgg(figure).draw()
        axis = figure.axes[0]
        self.assertEqual(len(axis.lines), 8)
        self.assertIn('Stage 1', axis.lines[0].get_label())
        self.assertIn('Stage 2', axis.lines[1].get_label())
        np.testing.assert_allclose(np.asarray(axis.lines[0].get_data_3d()).T,
                                   first_stage_eci_positions(solution) / 1000)
        for index, color in enumerate(('blue', 'red')):
            text = figure.axes[4 + index].texts[0]
            self.assertEqual(text.get_color(), color)
            np.testing.assert_allclose(text.get_bbox_patch().get_edgecolor(),
                                       (0., 0., 1., 1.) if color == 'blue' else (1., 0., 0., 1.))
            content = text.get_text()
            self.assertIn(f'Case {index + 1}', content)
            self.assertIn('12.500 t' if index == 0 else '15.000 t', content)
            self.assertIn('MECO: 170.0 s', content)
            self.assertIn('SECO: 520.0 s', content)
            self.assertIn('Max-Q (S1): 29.5 kPa', content)
            self.assertIn('Orbit-burn propellant: 1.200 t', content)
        np.testing.assert_array_equal(solution['x1'], original['x1'])
        draw_trajectory_3d(figure, cases[1:])
        self.assertEqual(len(figure.axes), 5)
        self.assertIn('Case 2', figure.axes[4].texts[0].get_text())

    def test_results_delta_v_matches_actual_speed_change_at_apogee(self):
        solution, target = elliptic_case()
        solution.update(payload_mass=12500., LaunchAz=np.array([np.pi/2]))
        parking, final = solution_orbits(solution, target)
        a = (parking['apogee'] + parking['perigee']) / 2
        speed = np.sqrt(VLType.mu*(2/parking['apogee'] - 1/a))
        expected_delta_v = abs(solution['v3_desired'] - speed)
        text = case_results_text(solution, 'Case 1', parking, final)
        self.assertIn(f'Δv at parking apogee: {expected_delta_v:,.2f} m/s', text)
        self.assertIn('Azimuth: 90.00 ° E of N', text)
        self.assertIn('Inclination: 53.00°', text)
        self.assertIn('Parking: 500.0 × 150.0 km', text)
        self.assertIn('Final: 500.0 × 200.0 km', text)
        missing = case_results_text({}, 'Case 1')
        self.assertIn('Payload: —', missing)
        self.assertIn('Δv / parking / final orbit: —', missing)

    def test_empty_view_and_unbound_state_keep_the_tab_usable(self):
        figure = Figure(figsize=(12, 8), constrained_layout=True)
        draw_trajectory_3d(figure, [])
        self.assertEqual(len(figure.axes), 4)
        self.assertIn('Run a case', figure.texts[0].get_text())
        solution, target = elliptic_case()
        solution['x3'][3:6] *= 2
        draw_trajectory_3d(figure, [(solution, target, 'blue', 'Case 1')])
        FigureCanvasAgg(figure).draw()
        self.assertEqual(len(figure.axes[0].lines), 1)
        self.assertIn('bound orbit', figure.texts[0].get_text())

    def test_gui_tab_respects_display_selection_and_skips_scalar_telemetry(self):
        self.assertIn('3D trajectory', RESULT_TABS)
        solution, target = elliptic_case()
        solution['payload_mass'] = 10000.
        app = OptimizationApp.__new__(OptimizationApp)
        app.cases = [ComparisonCase('Case 1', 'blue'), ComparisonCase('Case 2', 'red')]
        for case in app.cases:
            case.result, case.request = solution, {'target_orbit': target, 'configuration': 1, 'recovery': 'EXP'}
        app.case_visible = [Mock(get=Mock(return_value=True)), Mock(get=Mock(return_value=False))]
        app.telemetry = Mock()
        app.telemetry.groups = {'primary': {}, 'stage1': {}, 'stage2': {}, 'booster': {}}
        app.telemetry_visible = Mock(get=Mock(return_value=True))
        app.telemetry_primary = Mock(get=Mock(return_value='Stage 1 / booster'))
        app.telemetry_shift = 0.
        app.telemetry_info = Mock()
        app.notebook = Mock()
        app.tabs = {'Booster': Mock(), 'First stage': Mock()}
        app.figures = {'3D trajectory': Figure(figsize=(12, 8))}
        app.canvases = {'3D trajectory': Mock()}
        app.summary = Mock()
        app.update_case_labels = Mock()
        app.schedule_save = Mock()
        with patch('OptimizationGUI.overlay_telemetry') as overlay, \
             patch('OptimizationGUI.remaining_propellant_text', return_value='—'):
            app.refresh_comparison()
        overlay.assert_not_called()
        self.assertEqual(len(app.figures['3D trajectory'].axes[0].lines), 3)
        self.assertIn('Case 1', app.figures['3D trajectory'].axes[0].lines[0].get_label())
        app.canvases['3D trajectory'].draw_idle.assert_called_once()


if __name__ == '__main__':
    unittest.main()
