"""CSV data fidelity, historical snapshots and the GUI export action."""
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from LV_Type_scaled import DispesrionFactorsType
from OptimizationGUI import ComparisonCase, OptimizationApp, nominal_request
from PayloadSensitivity import compute_sensitivities
from SensitivityCSV import export_sensitivity_csv
from VehicleDefinitions import VEHICLE_PARAMETERS


class SensitivityCSVTests(unittest.TestCase):
    def request(self):
        return nominal_request(asdict(DispesrionFactorsType()), configuration=5,
            orbit_values={'apogee': '500', 'perigee': '300', 'inclination': '70'},
            launch_site={'name': 'Custom', 'latitude': 34., 'longitude': -120., 'altitude': 200.})

    def read_export(self, report):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.csv'
            export_sensitivity_csv(path, report)
            with path.open(encoding='utf-8-sig', newline='') as stream:
                return list(csv.DictReader(stream))

    def test_multiple_strategies_full_precision_and_run_time_inputs(self):
        def solve(request, init_guess=None):
            payload = (10000 if request['recovery'] == 'ASDS' else 8000) + request['dispersion']['SecondStageIsp'] * 123.456789012345
            return dict(success=True, payload_mass=np.array(payload), x1=np.ones((5, 3)),
                        configuration=request['configuration'])

        initial = dict(x1=np.arange(15).reshape(5, 3), payload_mass=np.array(9000.))
        request = self.request()
        request['recovery'] = 'ASDS'
        report = compute_sensitivities(request, {'SecondStageIsp': .0123456789012345},
            recoveries=['ASDS', 'RTLS'], solve=solve, initial_solution=initial)
        original = deepcopy(report)
        # Export must not consult the current built-in definition or the form.
        with patch('SensitivityCSV.VLType', side_effect=AssertionError('No export-time model lookup')):
            records = self.read_export(report)
        self.assertEqual(report, original)
        self.assertEqual(len(records), 4)
        self.assertEqual([row['recovery'] for row in records], ['ASDS', 'ASDS', 'RTLS', 'RTLS'])
        table = records[1]
        self.assertEqual(float(table['step']), report['rows'][0]['step'])
        self.assertEqual(float(table['derivative_kg_per_parameter_unit']), report['rows'][0]['derivative'])
        self.assertEqual(float(table['apogee_altitude_km']), 500.)
        self.assertEqual(float(table['launch_latitude_deg']), 34.)
        self.assertEqual(table['vehicle'], 'Falcon 9 True')
        for name in VEHICLE_PARAMETERS:
            self.assertNotEqual(table['nominal_si.' + name], '')
            self.assertNotEqual(table['effective_si.' + name], '')
        bundle = json.loads(records[0]['report_json'])
        self.assertEqual(bundle, report)
        snapshot = bundle['runs'][0]['reproducibility']
        self.assertEqual(snapshot['initial_solution']['x1'], initial['x1'].tolist())
        self.assertIn('LV_Optimization_Type.py', snapshot['source_sha256'])
        self.assertEqual(snapshot['baseline_solution']['x1'], np.ones((5, 3)).tolist())
        self.assertTrue(all(not row['report_json'] for row in records[1:]))

    def test_failed_baseline_and_failed_parameter_are_retained(self):
        def solve(request, init_guess=None):
            if request['recovery'] == 'RTLS' or request['dispersion']['SecondStageIsp'] > 1:
                raise RuntimeError('Failed, solver\nreason')
            return dict(success=True, payload_mass=10000.)
        report = compute_sensitivities(self.request(), {'SecondStageIsp': .01},
                                      recoveries=['ASDS', 'RTLS'], solve=solve)
        records = self.read_export(report)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]['status'], 'Failed')
        self.assertEqual(records[1]['plus_payload_kg'], '')
        self.assertIn('Failed, solver\nreason', json.loads(records[1]['errors_json'])['plus'])
        self.assertIn('Failed, solver\nreason', json.loads(records[2]['errors_json'])['baseline'])
        self.assertEqual(json.loads(records[0]['report_json']), report)

    def test_physical_parameter_units_and_nominal_vs_effective_vehicle(self):
        request = self.request()
        request['dispersion']['FirstStageThrust'] = .9
        report = compute_sensitivities(request, {'FirstStage_MaxDynamicPressure': 300.},
            solve=lambda candidate, init_guess=None: dict(success=True, payload_mass=10000.))
        records = self.read_export(report)
        self.assertEqual(records[1]['parameter_unit'], 'Pa')
        self.assertEqual(float(records[1]['step']), 300.)
        nominal = float(records[1]['nominal_si.FirstStage_SL_Thrust'])
        effective = float(records[1]['effective_si.FirstStage_SL_Thrust'])
        self.assertAlmostEqual(effective, .9 * nominal)
        self.assertIsInstance(report['solver_request']['configuration'], dict)

    def test_legacy_and_cancelled_reports_export_without_inventing_nominal_values(self):
        request = self.request()
        report = dict(request=request, rows=[], steps={'SecondStageIsp': .01},
                      termination='Cancelled', baseline=None)
        records = self.read_export(report)
        self.assertEqual(len(records), 1)
        self.assertIn('Legacy report', records[0]['reproducibility_note'])
        self.assertFalse(any(key.startswith('nominal_si.') for key in records[0]))
        self.assertEqual(json.loads(records[0]['report_json']), report)

    def test_gui_exports_selected_saved_report_and_handles_cancel_and_errors(self):
        app = OptimizationApp.__new__(OptimizationApp)
        app.root = Mock()
        app.selected_case = 1
        app.cases = [ComparisonCase('A', 'red'), ComparisonCase('B', 'blue')]
        app.cases[1].sensitivity_report = {'request': self.request(), 'rows': []}
        with patch('tkinter.filedialog.asksaveasfilename', return_value='results.csv') as dialog, \
             patch('tkinter.messagebox.showerror') as error, \
             patch('OptimizationGUI.export_sensitivity_csv') as export:
            app.export_sensitivity_results()
            export.assert_called_once_with('results.csv', app.cases[1].sensitivity_report)
            dialog.return_value = ''
            app.export_sensitivity_results()
            self.assertEqual(export.call_count, 1)
            dialog.return_value = 'results.csv'
            export.side_effect = OSError('Permission denied')
            app.export_sensitivity_results()
            error.assert_called_once()
        app.cases[1].sensitivity_report = None
        with patch('tkinter.messagebox.showinfo') as info, \
             patch('tkinter.filedialog.asksaveasfilename') as dialog:
            app.export_sensitivity_results()
            info.assert_called_once()
            dialog.assert_not_called()


if __name__ == '__main__':
    unittest.main()
