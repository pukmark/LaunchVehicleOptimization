"""Flight legends identify only main vehicle and booster when both are present."""
import unittest
from matplotlib.figure import Figure
from PlotLegends import set_vehicle_legend


class PlotLegendTests(unittest.TestCase):
    def test_axis_legend_collapses_phases_and_is_removed_for_a_single_group(self):
        axis = Figure().add_subplot()
        for label in ('Case 1 · Stage 1', 'Case 1 · Stage 2', 'Telemetry'):
            axis.plot([0, 1], [0, 1], label=label)
        axis.legend()
        set_vehicle_legend(axis, {'Main vehicle'})
        self.assertIsNone(axis.get_legend())
        set_vehicle_legend(axis, ['Main vehicle', 'Main vehicle', 'Booster'])
        self.assertEqual([text.get_text() for text in axis.get_legend().texts], ['Main vehicle', 'Booster'])
        set_vehicle_legend(axis, {'Booster'})
        self.assertIsNone(axis.get_legend())

    def test_figure_legend_is_replaced_without_leaving_old_case_entries(self):
        figure = Figure()
        figure.add_subplot().plot([0, 1], [0, 1], label='Case 1')
        figure.legend()
        set_vehicle_legend(figure, {'Main vehicle', 'Booster'})
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual([text.get_text() for text in figure.legends[0].texts], ['Main vehicle', 'Booster'])
        set_vehicle_legend(figure, {'Main vehicle'})
        self.assertFalse(figure.legends)
