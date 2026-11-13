"""CSV preserves independently sampled cases, boundaries, and telemetry."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from TelemetryCSV import load_telemetry_csv
from TrajectoryCSV import export_trajectory_csv


class TrajectoryCSVTests(unittest.TestCase):
    def parameters(self, path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        header = next(i for i, row in enumerate(rows) if row and row[0] == 'time_since_launch_s')
        self.assertEqual(rows[0], ['# Case', 'Parameter', 'Value'])
        self.assertEqual(rows[header - 1], [])
        parameters = [(row[0][2:], row[1], row[2]) for row in rows[1:header - 1]]
        self.assertEqual(len(parameters), len({(case, key) for case, key, _ in parameters}))
        return {(case, key): value for case, key, value in parameters}

    def solution(self):
        return {'t1_vec': np.array([0., 1., 1., 2.]),
                'x1': np.array([[0, 1, 2, 3], [0, 10, 20, 30], [0, 3, 4, 6],
                                 [0, 4, 3, 8], [100, 99, 98, 97]]),
                'alt1_sol': np.array([0, 10, 20, 30]),
                'Qdyn1_sol': np.array([0, 100, 200, 300]),
                'acc1_sol': np.array([1, 2, 3]), 'Isp1_sol': np.array([280, 281, 282]),
                'u1': np.array([[.5, .6, .7], [0, .1, .2]])}

    def test_cases_preserve_duplicate_times_controls_and_outer_join(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.csv'
            other = self.solution()
            other['t1_vec'] = np.array([0., .5, 1., 3.])
            rows, columns = export_trajectory_csv(path, [('Case 1', self.solution(), None), ('Case 3', other, None)])
            data = pd.read_csv(path, comment='#')
            self.assertEqual(data.columns[0], 'time_since_launch_s')
            self.assertEqual(rows, 6)
            self.assertEqual(columns, len(data.columns))
            np.testing.assert_array_equal(data['time_since_launch_s'], [0, .5, 1, 1, 2, 3])
            np.testing.assert_allclose(data['Case 1.stage1.mass_kg'].dropna(), [100, 99, 98, 97])
            np.testing.assert_allclose(data['Case 1.stage1.Qdyn_pa'].dropna(), [0, 100, 200, 300])
            np.testing.assert_allclose(data['Case 1.stage1.specific_acceleration_mps2'].dropna(), [1, 2, 3])
            self.assertTrue(pd.isna(data.loc[4, 'Case 1.stage1.thrust_factor']))
            self.assertFalse(any('Case 2' in key for key in data))

    def test_telemetry_only_shift_raw_extra_columns_and_no_source_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'telemetry.csv'
            source.write_text('time_s,speed_kmh,mass_kg,extra_note\n2,36,100,a\n0,0,102,b\n2,72,99,c\n')
            telemetry = load_telemetry_csv(source)
            source.unlink()
            output = Path(directory) / 'out.csv'
            export_trajectory_csv(output, telemetry=telemetry, telemetry_shift=10.)
            data = pd.read_csv(output, comment='#')
            np.testing.assert_array_equal(data.iloc[:, 0], [10, 12, 12])
            np.testing.assert_array_equal(data['telemetry.original.extra_note'], ['b', 'a', 'c'])
            np.testing.assert_array_equal(data['telemetry.primary.speed_mps'], [0, 10, 20])
            np.testing.assert_array_equal(telemetry.time, [0, 2, 2])

    def test_burn_uses_physical_time(self):
        burn = {'time': np.array([100., 101.]), 'physical_start': 900.,
                'state': np.array([[6378137, 6378137], [0, 0], [0, 0], [0, 0],
                                   [7000, 7001], [0, 0], [100, 99]]),
                'altitude': np.array([0., 0.]), 'thrust': np.ones((3, 2)), 'isp': 300}
        with tempfile.TemporaryDirectory() as directory, patch('TrajectoryCSV.second_burn_history', return_value=burn):
            path = Path(directory) / 'out.csv'
            export_trajectory_csv(path, [('Case 1', {}, None)])
            np.testing.assert_array_equal(pd.read_csv(path, comment='#').iloc[:, 0], [900, 901])

    def test_empty_selection_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.csv'
            with self.assertRaises(ValueError):
                export_trajectory_csv(path)
            self.assertFalse(path.exists())

    def test_saved_metadata_units_non_nominal_values_and_case_isolation(self):
        solution = dict(self.solution(), payload_mass=12345., configuration=1,
                        launch_site={'name': 'Custom', 'latitude': 12., 'longitude': -34., 'altitude': 50.},
                        dispersion={'FirstStageThrust': 1.05, 'FirstStageIsp': 1.,
                                    'FirstStage_EmptyMass': -100., 'LaunchAltDelta': 0.})
        target = {'Name': 'Custom', 'apogee': 6378137. + 420000.,
                  'perigee': 6378137. + 410000., 'i': np.deg2rad(51.6)}
        request = {'configuration': 2, 'payload_mass_predefined': 999.,
                   'launch_site': {'name': 'Equator', 'latitude': 0., 'longitude': 0., 'altitude': 0.},
                   'dispersion': {'FirstStageThrust': 1., 'FirstStageIsp': 1.},
                   'target_orbit': target}
        other = dict(self.solution(), payload_mass=500., configuration={'name': 'Saved custom vehicle'},
                     dispersion={'FirstStageThrust': 1., 'FirstStage_EmptyMass': 0.})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.csv'
            export_trajectory_csv(path, [('Case 1', solution, target), ('Case 3', other, None)],
                                  case_requests={'Case 1': request})
            data = pd.read_csv(path, comment='#')
            parameters = self.parameters(path)
            self.assertFalse(any('.metadata.' in column for column in data))
            self.assertEqual(data.columns[0], 'time_since_launch_s')
            for key, expected in {'payload_mass_kg': 12345., 'launch_vehicle_name': 'Falcon 9',
                                  'launch_site_name': 'Custom', 'launch_site_latitude_deg': 12.,
                                  'launch_site_longitude_deg': -34., 'launch_site_altitude_m': 50.,
                                  'target_apogee_altitude_km': 420., 'target_perigee_altitude_km': 410.,
                                  'target_inclination_deg': 51.6, 'dispersion.FirstStageThrust': 1.05,
                                  'dispersion.FirstStage_EmptyMass': -100.}.items():
                actual = parameters['Case 1', key]
                self.assertEqual(float(actual) if isinstance(expected, float) else actual, expected, key)
            self.assertEqual(json.loads(parameters['Case 1', 'non_nominal_dispersions_json']),
                             {'FirstStageThrust': 1.05, 'FirstStage_EmptyMass': -100.})
            self.assertNotIn(('Case 1', 'dispersion.FirstStageIsp'), parameters)
            self.assertNotIn(('Case 1', 'dispersion.LaunchAltDelta'), parameters)
            self.assertEqual(parameters['Case 3', 'non_nominal_dispersions_json'], '{}')
            self.assertEqual(parameters['Case 3', 'launch_vehicle_name'], 'Saved custom vehicle')
            self.assertEqual(parameters['Case 3', 'target_apogee_altitude_km'], '')
            self.assertFalse(any(case == 'Case 2' for case, _ in parameters))
            self.assertFalse(any(column.startswith('Case 2.') for column in data))

    def test_metadata_falls_back_to_saved_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.csv'
            export_trajectory_csv(path, [('Case 1', self.solution(), None)], case_requests={
                'Case 1': {'configuration': 3, 'dispersion': {'StagePartitionDelta': .02},
                           'target_orbit': {'apogee': 6578137., 'perigee': 6578137., 'i': 0.}}})
            data = pd.read_csv(path, comment='#')
            parameters = self.parameters(path)
            self.assertEqual(parameters['Case 1', 'launch_vehicle_name'], 'Starship V3')
            self.assertEqual(float(parameters['Case 1', 'dispersion.StagePartitionDelta']), .02)
            self.assertEqual(float(parameters['Case 1', 'target_apogee_altitude_km']), 200.)
            self.assertEqual(parameters['Case 1', 'payload_mass_kg'], '')
