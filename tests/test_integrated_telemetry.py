"""Measured channels on the combined mission timeline."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from IntegratedResults import draw_integrated_results, integrated_telemetry_curves
from TelemetryCSV import load_telemetry_csv


class IntegratedTelemetryTests(unittest.TestCase):
    def load(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'telemetry.csv'
        path.write_text(text)
        return load_telemetry_csv(path)

    def test_telemetry_only_all_five_channels_units_shift_and_shared_zoom(self):
        data = self.load('time_s,stage2_altitude_m,stage2_speed_kmh,stage2_mass_kg,'
                         'stage2_throttle_percent,stage2_angle_of_attack_deg\n'
                         '0,100000,3600,80000,65,4\n2,102000,7200,79000,70,5\n')
        figure = Figure(figsize=(14, 8), constrained_layout=True)
        self.assertTrue(draw_integrated_results(figure, [], telemetry=data,
                                               telemetry_shift=10., telemetry_color='purple'))
        FigureCanvasAgg(figure).draw()
        for axis, expected in zip(figure.axes, ([100, 102], [1000, 2000], [80, 79], [65, 70], [4, 5])):
            line = axis.lines[0]
            np.testing.assert_allclose(line.get_xdata(), [10, 12])
            np.testing.assert_allclose(line.get_ydata(), expected)
            self.assertEqual(line.get_color(), 'purple')
            self.assertIn('Stage 2', line.get_label())
        figure.axes[0].set_xlim(10., 11.)
        for axis in figure.axes:
            np.testing.assert_allclose(axis.get_xlim(), [10., 11.])

    def test_numbered_concurrent_histories_preserve_gaps_and_late_landing(self):
        data = self.load('time_s,altitude1_km,velocity1_kmh,altitude2_km,velocity2_kmh,'
                         'altitude3_km,velocity3_kmh\n'
                         '0,1,36,,,,\n1,,,2,72,1.5,18\n2,,,,,,\n3,,,3,108,0,0\n')
        curves = list(integrated_telemetry_curves(data, 'Phase columns',
                      [{'t1_vec': [0., 1.], 't4_vec': [1., 2.]}], shift=5.))
        altitude = {label: values for index, _, values, label, _ in curves if index == 0}
        self.assertEqual(len(altitude), 3)
        np.testing.assert_allclose(altitude['Telemetry · Stage 2'], [np.nan, 2, np.nan, 3], equal_nan=True)
        np.testing.assert_allclose(altitude['Telemetry · Booster return'], [np.nan, 1.5, np.nan, 0], equal_nan=True)
        self.assertTrue(all(np.array_equal(time, [5, 6, 7, 8]) for _, time, _, _, _ in curves))

    def test_shared_primary_routes_by_phase_but_explicit_samples_keep_full_extent(self):
        data = self.load('time_s,speed_mps,stage1_speed_mps,booster_speed_mps\n'
                         '0,10,,\n1,20,,\n2,30,,\n3,40,,\n4,50,99,5\n')
        solution = {'t1_vec': [0., 1.], 't4_vec': [2., 3.]}
        curves = list(integrated_telemetry_curves(data, 'Stage 1 / booster', [solution]))
        speed = {label: values for index, _, values, label, _ in curves if index == 1}
        np.testing.assert_allclose(speed['Telemetry · Stage 1'], [10, 20, np.nan, np.nan, 99], equal_nan=True)
        np.testing.assert_allclose(speed['Telemetry · Booster return'], [np.nan, np.nan, 30, 40, 5], equal_nan=True)
        curves = list(integrated_telemetry_curves(data, 'Stage 1 / booster', []))
        shared = [values for index, _, values, label, _ in curves
                  if index == 1 and label == 'Telemetry · Stage 1 / booster']
        self.assertEqual(len(shared), 1)
        np.testing.assert_allclose(shared[0], [10, 20, 30, 40, 99])

    def test_booster_angles_only_during_boostback_and_shifted_in_mission_time(self):
        data = self.load('time_s,booster_alpha_deg,booster_throttle\n'
                         '0,170,.5\n1,175,.4\n2,180,.3\n3,180,.2\n')
        solution = {'t1_vec': [0., 1.], 't4_vec': [10., 15.], 'dt4_boostback': 1.}
        curves = list(integrated_telemetry_curves(data, 'Phase columns', [solution], shift=10.))
        angle = next(values for index, _, values, _, _ in curves if index == 4)
        np.testing.assert_allclose(angle, [170, 175, np.nan, np.nan], equal_nan=True)
        self.assertFalse(any(index == 4 for index, *_ in integrated_telemetry_curves(
            data, 'Phase columns', [{'t1_vec': [0., 1.], 't4_vec': [10., 15.]}], shift=10.)))
        throttle = next(values for index, _, values, _, _ in curves if index == 3)
        np.testing.assert_allclose(throttle, [50, 40, 30, 20])

    def test_vector_speed_and_primary_assignment(self):
        data = self.load('time_s,ecef_vx_mps,ecef_vy_mps,ecef_vz_mps,speed_mps\n'
                         '0,3,4,0,99\n1,,,0,7\n')
        curves = list(integrated_telemetry_curves(data, 'Stage 2', []))
        speed = next(values for index, _, values, label, _ in curves if index == 1)
        np.testing.assert_allclose(speed, [5, 7])
        self.assertTrue(all('Stage 2' in label for _, _, _, label, _ in curves))

    def test_gui_telemetry_visibility_color_shift_and_overlay_count(self):
        from OptimizationGUI import OptimizationApp
        app = OptimizationApp.__new__(OptimizationApp)
        app.cases, app.case_visible = [], []
        app.telemetry = self.load('time_s,speed_mps\n0,10\n1,20\n')
        app.telemetry_visible = Mock(get=Mock(return_value=True))
        app.telemetry_primary = Mock(get=Mock(return_value='Stage 2'))
        app.telemetry_color, app.telemetry_shift = 'purple', 5.
        app.figures = {'Summery': Figure()}
        app.canvases = {'Summery': Mock()}
        app.tabs = {'Booster': Mock(), 'Booster 2': Mock(), 'First stage': Mock()}
        app.notebook, app.telemetry_info, app.summary = Mock(), Mock(), Mock()
        app.update_case_labels, app.render_sensitivity_report, app.schedule_save = Mock(), Mock(), Mock()
        app.refresh_comparison()
        line = app.figures['Summery'].axes[1].lines[0]
        np.testing.assert_allclose(line.get_xdata(), [5, 6])
        self.assertEqual(line.get_color(), 'purple')
        self.assertIn('1 tabs overlaid', app.telemetry_info.set.call_args.args[0])
        app.telemetry_visible.get.return_value = False
        app.refresh_comparison()
        self.assertEqual(len(app.figures['Summery'].axes), 0)


if __name__ == '__main__':
    unittest.main()
