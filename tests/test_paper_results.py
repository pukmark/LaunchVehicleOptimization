"""Paper comparison curves, reference frames, minimal legends, and exports."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from PaperResults import draw_paper_results, paper_comparison_costs
from TelemetryCSV import TelemetryData
from LV_Type_scaled import VLType


class PaperResultsTests(unittest.TestCase):
    def solution(self):
        first = np.zeros((5, 3))
        first[2] = [0, 100, 200]
        upper = np.zeros((7, 3))
        upper[0] = VLType.R0 + 100000
        upper[4] = VLType.omega_earth * upper[0] + 300
        booster = np.zeros((3, 5))
        booster[:, 2] = [200, 100, 0]
        return dict(t1_vec=np.array([0, 1, 2]), x1=first, alt1_sol=[0, 1000, 2000],
                    t2_vec=[2, 3, 4], x2=upper, alt2_sol=[2000, 3000, 4000],
                    t3_vec=[4, 5, 6], x3=upper, alt3_sol=[4000, 5000, 6000],
                    t4_vec=[2, 4, 6], x4=booster, alt4_sol=[2000, 1000, 0])

    def test_all_phases_ecef_speed_units_and_minimal_legend(self):
        figure = Figure(figsize=(7.2, 6.0))
        solution = self.solution()
        draw_paper_results(figure, [(solution, 'blue', 'Case 1'), (solution, 'red', 'Case 2')])
        FigureCanvasAgg(figure).draw()
        self.assertEqual(len(figure.axes), 3)
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0, 1, 2])
        np.testing.assert_allclose(figure.axes[1].lines[1].get_ydata(), 300.)
        self.assertEqual(len(figure.axes[0].lines), 8)
        self.assertEqual([t.get_text() for t in figure.legends[0].texts], ['Main vehicle', 'Booster'])
        self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
        with tempfile.TemporaryDirectory() as folder:
            for extension in ('pdf', 'svg', 'png'):
                path = Path(folder) / f'comparison.{extension}'
                figure.savefig(path, dpi=100)
                self.assertGreater(path.stat().st_size, 1000)

    def test_telemetry_full_extent_shift_gaps_and_single_legend_entry(self):
        time = np.arange(4.)
        groups = {'primary': {}, 'stage1': {'altitude': [0, 1, np.nan, 3]},
                  'stage2': {'speed': [200, 300, np.nan, 500]},
                  'booster': {'altitude': [3, 2, np.nan, 0]}}
        data = TelemetryData(Path('test.csv'), time, groups, 'test')
        figure = Figure()
        self.assertTrue(draw_paper_results(figure, [], telemetry=data, telemetry_shift=7., telemetry_color='black'))
        for index, line_index in ((0, 0), (1, 0), (0, 1)):
            line = figure.axes[index].lines[line_index]
            np.testing.assert_allclose(line.get_xdata(), time+7)
            self.assertTrue(np.isnan(line.get_ydata()[2]))
            self.assertEqual(line.get_color(), 'black')
        self.assertEqual([t.get_text() for t in figure.legends[0].texts], ['Main vehicle', 'Booster'])
        draw_paper_results(figure, [])
        self.assertEqual(len(figure.axes), 3)
        self.assertFalse(figure.legends)

    def test_main_vehicle_only_omits_legend_even_with_multiple_burns_and_cases(self):
        solution = self.solution()
        for key in ('x4', 't4_vec', 'alt4_sol'):
            del solution[key]
        figure = Figure()
        draw_paper_results(figure, [(solution, 'blue', 'Case 1'), (solution, 'red', 'Case 2')])
        self.assertFalse(figure.legends)
        self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))

    def test_second_stage_coast_is_included(self):
        figure = Figure()
        solution = self.solution()
        coast_state = np.asarray(solution['x2'])
        with patch('PaperResults.second_stage_coast', return_value=(np.array([2, 2.5, 3]), coast_state, np.array([2000, 2500, 3000]))):
            draw_paper_results(figure, [(solution, 'blue', 'Case 1')])
        self.assertEqual(len(figure.axes[0].lines), 5)
        np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(), [2, 2.5, 3])

    def test_speed_derivative_all_simulation_segments_and_ecef_conversion(self):
        solution = self.solution()
        # ECI velocity is constant here, while omega cross position changes.
        # Its ECEF speed derivative must therefore be nonzero.
        solution['x2'] = solution['x2'].copy()
        solution['x2'][0] += np.array([0., 1000., 2000.])
        figure = Figure()
        with patch('PaperResults.second_stage_coast', return_value=(
                np.array([2., 2.5, 3.]), solution['x3'], np.array([2000, 2500, 3000]))):
            draw_paper_results(figure, [(solution, 'blue', 'Case 1')])
        acceleration = figure.axes[2]
        self.assertEqual(len(acceleration.lines), 5)
        expected = (100., 0., -1000.*VLType.omega_earth, 0., -50.)
        for line, derivative in zip(acceleration.lines, expected):
            self.assertTrue(np.isnan(line.get_ydata()[0]))
            np.testing.assert_allclose(line.get_ydata()[1:], derivative, atol=1e-10)

    def test_telemetry_derivative_all_phases_preserves_gaps_and_ignores_accelerometer(self):
        time = np.array([0., 1., 3., 4., 5., 6.])
        speed = np.array([10., 20., 10., np.nan, 30., 34.])
        data = TelemetryData(Path('test.csv'), time,
                             dict(primary={}, **{phase: {'speed': speed*scale,
                                                         'acceleration': np.full(6, 999.)}
                                                 for phase, scale in [('stage1', 1), ('stage2', 2), ('booster', 3)]}),
                             'test')
        figure = Figure()
        draw_paper_results(figure, [], telemetry=data, telemetry_shift=7., telemetry_color='black')
        self.assertEqual(len(figure.axes[2].lines), 3)
        for scale, line in enumerate(figure.axes[2].lines, 1):
            np.testing.assert_allclose(line.get_xdata(), time+7.)
            np.testing.assert_allclose(line.get_ydata(),
                                       np.array([np.nan, 10., -5., np.nan, np.nan, 4.])*scale,
                                       equal_nan=True)
            self.assertEqual(line.get_color(), 'black')

    def test_gui_refresh_uses_displayed_cases_and_current_telemetry_settings(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        from unittest.mock import Mock
        from LV_Type_scaled import DispesrionFactorsType
        from OptimizationGUI import OptimizationApp, nominal_request
        app = OptimizationApp.__new__(OptimizationApp)
        request = nominal_request(asdict(DispesrionFactorsType()), payload_mass=10000.)
        shown = SimpleNamespace(name='Case 1', result=dict(self.solution(), payload_mass=10000.),
                                request=request, color='blue')
        hidden = SimpleNamespace(name='Case 2', result=self.solution(), color='red')
        app.cases, app.case_visible = [shown, hidden], [Mock(get=lambda: True), Mock(get=lambda: False)]
        app.telemetry = TelemetryData(Path('test.csv'), np.arange(2.),
                                      {'primary': {'speed': np.array([1., 2.])},
                                       'stage1': {}, 'stage2': {}, 'booster': {}}, 'test')
        app.telemetry_visible = Mock(get=Mock(return_value=True))
        app.telemetry_primary = Mock(get=lambda: 'Stage 2')
        app.telemetry_color, app.telemetry_shift = 'purple', 5.
        app.figures, app.canvases = {'Paper figures': Figure()}, {'Paper figures': Mock()}
        app.tabs = {'Booster': Mock(), 'Booster 2': Mock(), 'First stage': Mock()}
        app.notebook, app.telemetry_info, app.summary = Mock(), Mock(), Mock()
        app.update_case_labels, app.render_sensitivity_report, app.schedule_save = Mock(), Mock(), Mock()
        with patch('OptimizationGUI.draw_paper_results', return_value=True) as draw, \
             patch('OptimizationGUI.remaining_propellant_text', return_value='test'), \
             patch('OptimizationGUI.maneuver_summary_text', return_value='test'):
            app.refresh_comparison()
            self.assertEqual(draw.call_args.args[1], [(shown.result, 'blue', 'Case 1')])
            self.assertIs(draw.call_args.kwargs['telemetry'], app.telemetry)
            self.assertEqual(draw.call_args.kwargs['telemetry_shift'], 5.)
            app.telemetry_visible.get.return_value = False
            app.refresh_comparison()
            self.assertIsNone(draw.call_args.kwargs['telemetry'])

    def test_comparison_costs_normalization_weighting_shift_and_coverage(self):
        # Stage-one altitude residuals are 1 and 3 normalized units; speed
        # residuals are 2 and 2. Booster altitude residual is 4. Each channel
        # has equal weight within a phase, then each phase has equal weight.
        solution = self.solution()
        time = np.array([0., 1., 2., 6., 7.])
        groups = {'primary': {},
                  'stage1': {'altitude': np.array([-1., -2., np.nan, np.nan, 0.]),
                             'speed': np.array([-50., 50., np.nan, np.nan, np.nan])},
                  'stage2': {},
                  'booster': {'altitude': np.array([np.nan, np.nan, np.nan, -4., np.nan])}}
        data = TelemetryData(Path('test.csv'), time - 10., groups, 'test')
        result = paper_comparison_costs([(solution, 'blue', 'Case 1')], telemetry=data, telemetry_shift=10.)[0]
        robust = lambda r: 2*(np.sqrt(1+r*r)-1)
        self.assertAlmostEqual(result['squared'], ((5.+4.)/2.+16.)/2.)
        self.assertAlmostEqual(result['robust'], (((robust(1)+robust(3))/2.+robust(2))/2.+robust(4))/2.)
        self.assertEqual((result['matched'], result['observed']), (5, 6))
        self.assertEqual(paper_comparison_costs([], telemetry=data), [])
        self.assertEqual(paper_comparison_costs([(solution, 'blue', 'Case 1')]), [])
        empty = paper_comparison_costs([(solution, 'blue', 'Case 1')], telemetry=data, telemetry_shift=100.)[0]
        self.assertIsNone(empty['robust'])
        self.assertIsNone(empty['squared'])
        self.assertEqual(empty['matched'], 0)

    def test_gui_cost_readout_reports_both_losses(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from OptimizationGUI import OptimizationApp
        app = OptimizationApp.__new__(OptimizationApp)
        app.paper_cost_text, app.telemetry_shift = Mock(), 7.
        case = SimpleNamespace(result=self.solution(), color='blue', name='Case 1')
        data = TelemetryData(Path('test.csv'), np.array([-7., -6.]),
                             {'primary': {}, 'stage1': {'altitude': np.array([-3., -2.])},
                              'stage2': {}, 'booster': {}}, 'test')
        app.refresh_paper_costs([case], data, 'Phase columns')
        text = app.paper_cost_text.set.call_args.args[0]
        self.assertIn('Case 1', text)
        self.assertIn('Robust', text)
        self.assertIn('Squared', text)
        self.assertIn('2/2', text)
        self.assertIn('9', text)
        app.refresh_paper_costs([case], None, 'Phase columns')
        self.assertNotIn('Case 1', app.paper_cost_text.set.call_args.args[0])
