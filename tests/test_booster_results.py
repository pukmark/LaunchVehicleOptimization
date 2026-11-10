"""Booster histories use altitude as the shared vertical axis."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from BoosterResults import draw_booster_altitude_results
from TelemetryCSV import aerodynamic_loads, load_telemetry_csv, numerical_acceleration
from LV_Type_scaled import VLType


class BoosterAltitudeTests(unittest.TestCase):
    def solution(self):
        return dict(x4=np.array([[0, 100000, 100, 200, 30000],
                                 [100, 110000, 50, -100, 29000],
                                 [200, 1000, 0, -50, 28000]]),
                    t4_vec=np.array([10., 20., 30.]), alt4_sol=np.array([100000., 110000., 1000.]),
                    Qdyn4_sol=np.array([100., 200., 300.]), acc4_sol=np.array([1., 2., 3.]),
                    dispersion={'AtmosphereDensity': 2.})

    def load(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'telemetry.csv'
        path.write_text(text)
        return load_telemetry_csv(path)

    def test_simulation_units_shared_altitude_and_independent_horizontal_axes(self):
        solution = self.solution()
        figure = Figure(figsize=(14, 8), constrained_layout=True)
        self.assertFalse(draw_booster_altitude_results(figure, [(solution, 'blue', 'Case 1')]))
        FigureCanvasAgg(figure).draw()
        self.assertEqual(len(figure.axes), 5)
        self.assertIn('dv/dt', figure.axes[4].lines[0].get_label())
        expected = [np.linalg.norm(solution['x4'][:, 2:4], axis=1), [30, 29, 28], [.1, .2, .3],
                    aerodynamic_loads(solution['alt4_sol'], np.linalg.norm(solution['x4'][:, 2:4], axis=1),
                                      density_factor=2., heat_coefficient=VLType.Booster_k_empirical)['heat_flux'],
                    numerical_acceleration(solution['t4_vec'], np.linalg.norm(solution['x4'][:, 2:4], axis=1))]
        for axis, values in zip(figure.axes, expected):
            self.assertIsNone(axis.get_legend())
            self.assertEqual(axis.get_ylabel(), 'Altitude [km]')
            np.testing.assert_allclose(axis.lines[0].get_ydata(), [110, 1])
            np.testing.assert_allclose(axis.lines[0].get_xdata(),
                                       [np.nan, values[-1]] if axis is figure.axes[4] else np.asarray(values)[1:])
            self.assertEqual(axis.lines[0].get_linestyle(), '--')
        figure.axes[3].set_ylim(20., 40.)
        for axis in figure.axes:
            np.testing.assert_allclose(axis.get_ylim(), [20., 40.])
        figure.axes[0].set_xlim(0., 1000.)
        figure.axes[1].set_xlim(20., 35.)
        np.testing.assert_allclose(figure.axes[0].get_xlim(), [0., 1000.])

    def test_telemetry_only_measurements_units_and_derived_labels(self):
        data = self.load('time_s,booster_altitude_m,booster_speed_kmh,booster_mass_kg,'
                         'booster_dynamic_pressure_pa,booster_heat_flux_w_m2,booster_acceleration_mps2\n'
                         '0,10000,3600,30000,20000,100000,2\n1,9000,3240,29000,21000,90000,3\n')
        figure = Figure()
        self.assertTrue(draw_booster_altitude_results(figure, [], telemetry=data, telemetry_color='purple'))
        for axis, expected in zip(figure.axes, ([1000, 900], [30, 29], [20, 21], [100, 90], [np.nan, -100])):
            np.testing.assert_allclose(axis.lines[0].get_xdata(), expected)
            np.testing.assert_allclose(axis.lines[0].get_ydata(), [10, 9])
            self.assertEqual(axis.lines[0].get_color(), 'purple')
        self.assertTrue(any('derived' in line.get_label() for line in figure.axes[3].lines))
        self.assertTrue(any('dv/dt' in line.get_label() for line in figure.axes[4].lines))

    def test_booster_plots_compare_signed_speed_derivatives_and_ignore_specific_acceleration(self):
        from OptimizationGUI import draw_phase
        from TelemetryCSV import overlay_telemetry
        solution = self.solution()
        solution['x4'][:, 2] = [100., 60., 80.]
        solution['x4'][:, 3] = 0.
        # Deliberately different proper acceleration must never appear here.
        solution['acc4_sol'] = np.full(3, 99.)
        data = self.load('time_s,booster_altitude_km,booster_speed_mps,booster_acceleration_mps2\n'
                         '10,100,100,999\n20,110,60,999\n30,1,80,999\n')
        for altitude_plot in (False, True):
            with self.subTest(altitude_plot=altitude_plot):
                figure = Figure()
                if altitude_plot:
                    draw_booster_altitude_results(figure, [(solution, 'blue', 'Case')], telemetry=data)
                else:
                    draw_phase(figure, 'Booster', solution, {})
                    overlay_telemetry(figure, 'Booster', data, 'Phase columns', 'purple')
                axis = figure.axes[4]
                self.assertEqual(len(axis.lines), 2)
                self.assertIn('includes gravity', axis.get_title())
                for line in axis.lines:
                    self.assertIn('dv/dt', line.get_label())
                    values = line.get_xdata() if altitude_plot else line.get_ydata()
                    expected = ([np.nan, 2.] if altitude_plot and line.get_label().startswith('Case')
                                else [np.nan, np.nan, 2.] if altitude_plot else [np.nan, -4., 2.])
                    np.testing.assert_allclose(values, expected, equal_nan=True)

    def test_shared_feed_time_shift_and_explicit_booster_samples_beyond_window(self):
        data = self.load('time_s,altitude_km,speed_mps,booster_altitude_km,booster_speed_mps\n'
                         '0,10,100,,\n1,9,90,,\n2,8,80,,\n3,7,70,6,60\n')
        solution = self.solution()
        solution['t4_vec'] = np.array([11., 11.5, 12.])
        figure = Figure()
        draw_booster_altitude_results(figure, [(solution, 'blue', 'Case 1')], telemetry=data,
                                      primary_group='Stage 1 / booster', telemetry_shift=10.)
        line = figure.axes[0].lines[1]
        np.testing.assert_allclose(line.get_xdata(), [np.nan, 90, 80, 60], equal_nan=True)
        np.testing.assert_allclose(line.get_ydata(), [np.nan, 9, 8, 6], equal_nan=True)

    def test_rtls_uses_ballistic_apogee_after_coast_and_boostback(self):
        solution = self.solution()
        time = np.arange(8.)
        solution.update(t4_vec=time, alt4_sol=np.array([100., 200., 150., 160., 180., 170., 150., 0.]),
                        x4=np.column_stack((time, time, time+1, time*0, 30000-time)),
                        Qdyn4_sol=time*1000, dt4_coast_after_sep=1., dt4_boostback=1., dt4_ballistic=3.)
        figure = Figure()
        data = self.load('time_s,booster_altitude_m,booster_speed_mps\n' + ''.join(
            f'{t},{a},{t+1}\n' for t, a in zip(time, solution['alt4_sol'])))
        draw_booster_altitude_results(figure, [(solution, 'blue', 'RTLS')], telemetry=data)
        # The higher pre-boostback peak at t=1 is excluded from the search.
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [5, 6, 7, 8])
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [.180, .170, .150, 0.])
        np.testing.assert_allclose(figure.axes[4].lines[0].get_xdata(), [np.nan, 1, 1, 1], equal_nan=True)
        np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(),
                                   [np.nan]*4 + [.180, .170, .150, 0.], equal_nan=True)

    def test_each_case_and_telemetry_start_at_their_own_apogee(self):
        first, second = self.solution(), self.solution()
        second['alt4_sol'] = np.array([120000., 110000., 1000.])
        data = self.load('time_s,booster_altitude_km,booster_speed_mps\n'
                         '0,10,100\n1,20,90\n2,15,80\n3,1,70\n')
        figure = Figure()
        draw_booster_altitude_results(figure, [(first, 'blue', 'First'), (second, 'red', 'Second')],
                                      telemetry=data, telemetry_shift=10.)
        lines = figure.axes[0].lines
        np.testing.assert_allclose(lines[0].get_ydata(), [110, 1])
        np.testing.assert_allclose(lines[1].get_ydata(), [120, 110, 1])
        np.testing.assert_allclose(lines[2].get_ydata(), [np.nan, 20, 15, 1], equal_nan=True)
        acceleration = figure.axes[4].lines[-1].get_xdata()
        np.testing.assert_allclose(acceleration, [np.nan, np.nan, -10, -10], equal_nan=True)

    def test_tab_order_and_gui_booster_visibility(self):
        from OptimizationGUI import OptimizationApp, RESULT_TABS
        self.assertEqual(RESULT_TABS[0], 'Summery')
        self.assertLess(RESULT_TABS.index('Booster'), RESULT_TABS.index('Booster 2'))
        app = OptimizationApp.__new__(OptimizationApp)
        app.cases, app.case_visible = [], []
        app.telemetry = self.load('time_s,booster_altitude_km,booster_speed_mps\n0,10,100\n1,9,90\n')
        app.telemetry_visible = Mock(get=Mock(return_value=True))
        app.telemetry_primary = Mock(get=Mock(return_value='Phase columns'))
        app.telemetry_color, app.telemetry_shift = 'purple', 0.
        app.figures, app.canvases = {'Booster 2': Figure()}, {'Booster 2': Mock()}
        app.tabs = {name: Mock() for name in ('Booster', 'Booster 2', 'First stage')}
        app.notebook, app.telemetry_info, app.summary = Mock(), Mock(), Mock()
        app.update_case_labels, app.render_sensitivity_report, app.schedule_save = Mock(), Mock(), Mock()
        app.refresh_comparison()
        app.notebook.add.assert_any_call(app.tabs['Booster 2'], text='Booster 2')
        self.assertTrue(app.figures['Booster 2'].axes[0].lines)
        app.telemetry_visible.get.return_value = False
        app.refresh_comparison()
        app.notebook.hide.assert_any_call(app.tabs['Booster 2'])
        self.assertFalse(app.figures['Booster 2'].axes[0].lines)


if __name__ == '__main__':
    unittest.main()
