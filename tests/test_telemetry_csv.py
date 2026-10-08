"""Telemetry parsing and overlays, including units, gaps, time, and stage routing."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from matplotlib.figure import Figure

from TelemetryCSV import aerodynamic_loads, load_telemetry_csv, mission_seconds, overlay_telemetry, interpolate_speed_time, numerical_acceleration


class TelemetryTests(unittest.TestCase):
    def load(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "telemetry.csv"
        path.write_text(text)
        return load_telemetry_csv(path)

    def test_signed_mission_time_and_invalid_values(self):
        self.assertEqual(mission_seconds("T-00:00:03.5"), -3.5)
        self.assertEqual(mission_seconds("T+01:02:03"), 3723)
        self.assertTrue(np.isnan(mission_seconds("00:61:00")))
        self.assertTrue(np.isnan(mission_seconds("bad")))

    def test_numbered_three_phase_schema_units_gaps_and_concurrent_return(self):
        data = self.load('t_video_sec,time_str_primary,velocity1_kmh,altitude1_km,acceleration1_g,'
                         'velocity2_kmh,altitude2_km,velocity3_kmh,altitude3_km\n'
                         '10,T+00:00:00,36,0.1,1,,,,\n'
                         '11,T+00:00:01,72,0.2,2,,,,\n'
                         '12,T+00:00:02,,,,360,3,180,2\n'
                         '13,T+00:00:03,,,,720,4,,\n')
        self.assertEqual(data.groups['primary'], {})
        np.testing.assert_allclose(data.time, [0, 1, 2, 3])
        for mode in ('Phase columns', 'Stage 1 / booster', 'Stage 2', 'Booster'):
            with self.subTest(mode=mode):
                first = data.channels('stage1', mode)
                upper = data.channels('stage2', mode)
                booster = data.channels('booster', mode)
                np.testing.assert_allclose(first['speed'], [10, 20, np.nan, np.nan], equal_nan=True)
                np.testing.assert_allclose(first['altitude'], [.1, .2, np.nan, np.nan], equal_nan=True)
                np.testing.assert_allclose(first['acceleration'], [9.80665, 19.6133, np.nan, np.nan], equal_nan=True)
                np.testing.assert_allclose(upper['speed'], [np.nan, np.nan, 100, 200], equal_nan=True)
                np.testing.assert_allclose(upper['altitude'], [np.nan, np.nan, 3, 4], equal_nan=True)
                np.testing.assert_allclose(booster['speed'], [np.nan, np.nan, 50, np.nan], equal_nan=True)
                np.testing.assert_allclose(booster['altitude'], [np.nan, np.nan, 2, np.nan], equal_nan=True)
        figure = Figure()
        self.assertTrue(overlay_telemetry(figure, 'Booster', data, 'Phase columns', 'purple'))
        np.testing.assert_allclose(figure.axes[2].lines[0].get_ydata(),
                                   [np.nan, np.nan, 50, np.nan], equal_nan=True)
        figure = Figure()
        self.assertTrue(overlay_telemetry(figure, 'S1 details', data, 'Phase columns', 'purple'))
        np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(),
                                   [9.80665, 19.6133, np.nan, np.nan], equal_nan=True)

    def test_numbered_phase_columns_prefer_new_values_and_support_exp(self):
        data = self.load('time_s,velocity1_kmh,stage1_speed_mps,altitude1_km,acceleration1_g,'
                         'velocity2_kmh,altitude2_km,velocity3_kmh,altitude3_km\n'
                         '0,36,99,0,1,,,,\n1,,,,,720,3,,\n')
        self.assertEqual(data.groups['stage1']['speed'][0], 10.)
        self.assertEqual(data.groups['booster'], {})
        self.assertEqual(data.groups['primary'], {})
        self.assertNotIn('acceleration', data.groups['stage2'])

    def test_clean_schema_units_gaps_and_separate_stage_two(self):
        data = self.load("time_s,speed_kmh,altitude_km,acceleration_g,velocity2_kmh,altitude2_km\n"
                         "2,360,1,2,720,3\n0,0,0,1,,\n1,,0.5,,,\n")
        np.testing.assert_allclose(data.time, [0, 1, 2])
        np.testing.assert_allclose(data.groups['primary']['speed'], [0, np.nan, 100], equal_nan=True)
        np.testing.assert_allclose(data.groups['primary']['acceleration'], [9.80665, np.nan, 19.6133], equal_nan=True)
        self.assertEqual(data.groups['stage2']['speed'][-1], 200)
        self.assertEqual(data.groups['stage2']['altitude'][-1], 3)

    def test_raw_schema_fallback_preserves_sign_and_aligns_video_time(self):
        data = self.load("t_video_sec,time_str_primary,vel_kmh_speed,vel_kmh_primary,alt_km_altitude_roi,accel_g_primary\n"
                         "100,T-00:00:02,,36,1,1\n101,,72,,2,2\n102,T+00:00:00,108,,3,3\n")
        np.testing.assert_allclose(data.time, [-2, -1, 0])
        np.testing.assert_allclose(data.groups['primary']['speed'], [10, 20, 30])

    def test_video_only_time_and_prefixed_channels(self):
        data = self.load("t_video_sec,stage1_altitude_m,s2_thrust_n,booster_mass_kg\n100,1000,500000,30000\n102,2000,600000,29000\n")
        np.testing.assert_allclose(data.time, [0, 2])
        self.assertIn('Video-relative', data.time_note)
        np.testing.assert_allclose(data.groups['stage1']['altitude'], [1, 2])
        np.testing.assert_allclose(data.groups['stage2']['thrust'], [500, 600])
        np.testing.assert_allclose(data.groups['booster']['mass'], [30, 29])

    def test_primary_destination_overlay_shift_and_phase_window(self):
        data = self.load("time_s,altitude_km,speed_mps\n0,0,0\n1,1,10\n2,,20\n3,3,30\n")
        figure = Figure()
        self.assertTrue(overlay_telemetry(figure, 'First stage', data, 'Stage 1 / booster', 'purple', 5, [(5, 7)]))
        np.testing.assert_allclose(figure.axes[1].lines[0].get_xdata(), [5, 6, 7, 8])
        np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [0, 1, np.nan, np.nan], equal_nan=True)
        self.assertFalse(overlay_telemetry(Figure(), 'Second stage', data, 'Stage 1 / booster', 'purple'))
        self.assertTrue(overlay_telemetry(Figure(), 'Second stage', data, 'Stage 2', 'purple'))
        self.assertEqual(figure.axes[1].lines[0].get_color(), 'purple')

    def test_invalid_or_empty_measurements_are_rejected(self):
        for content in ("time_s,altitude_km\n", "time_s,altitude_km\nbad,3\n", "time_s,unknown\n0,1\n", "time_s,speed_kmh\n0,inf\n"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.load(content)

    def test_derived_loads_units_scaling_and_missing_samples(self):
        nominal = aerodynamic_loads([0, 0, np.nan, 1000, 0], [100, 200, 100, np.nan, -5])
        np.testing.assert_allclose(nominal["pressure"][:2], [6.125, 24.5])
        np.testing.assert_allclose(nominal["heat_flux"][:2],
                                   2e-4 * np.sqrt(1.225) * np.array([100, 200])**3 / 1000)
        self.assertTrue(np.isnan(nominal["pressure"][2:]).all())
        self.assertTrue(np.isnan(nominal["heat_flux"][2:]).all())
        denser = aerodynamic_loads([0, 0], [100, 200], density_factor=4)
        np.testing.assert_allclose(denser["pressure"], 4 * nominal["pressure"][:2])
        np.testing.assert_allclose(denser["heat_flux"], 2 * nominal["heat_flux"][:2])

    def test_derived_loads_match_return_model(self):
        from Atmosphere_Type import AtmosphereType
        from LV_Type_scaled import BoosterReturn_2D, DispesrionFactorsType
        model = BoosterReturn_2D(AtmosphereType(), 1, DispesrionFactorsType(AtmosphereDensity=1.44))
        state = [0, 10000, 100, -300, 30000]
        loads = aerodynamic_loads([10000], [np.hypot(100, 300)], density_factor=1.44)
        self.assertAlmostEqual(loads["pressure"][0], float(model.dynamic_pressure_fun(model.scale_x(state))) / 1000)
        self.assertAlmostEqual(loads["heat_flux"][0], float(model.heat_flux_fun(model.scale_x(state))) / 1000)

    def test_telemetry_loads_are_derived_after_stage_routing_and_keep_csv_readings(self):
        data = self.load("time_s,altitude_km,speed_kmh,booster_altitude_km,booster_speed_mps,booster_dynamic_pressure_kpa,booster_heat_flux_w_m2\n"
                         "0,0,360,0,200,99,1000\n1,1,,2,,,\n")
        first = data.channels('stage1', 'Stage 1 / booster')
        booster = data.channels('booster', 'Stage 1 / booster')
        self.assertAlmostEqual(first['pressure_derived'][0], 6.125)
        self.assertAlmostEqual(booster['pressure_derived'][0], 24.5)
        self.assertEqual(booster['pressure'][0], 99)
        self.assertEqual(booster['heat_flux'][0], 1)
        self.assertTrue(np.isnan(booster['heat_flux_derived'][1]))
        self.assertNotIn('pressure_derived', data.channels('stage2', 'Stage 2'))
        figure = Figure()
        self.assertTrue(overlay_telemetry(figure, 'Booster', data, 'Stage 1 / booster', 'purple'))
        self.assertEqual(len(figure.axes), 7)
        self.assertEqual(len(figure.axes[5].lines), 2)
        self.assertEqual(len(figure.axes[6].lines), 2)
        self.assertEqual(figure.axes[6].lines[1].get_label(), 'Telemetry · derived')
        np.testing.assert_allclose(figure.axes[6].lines[1].get_ydata(), booster['heat_flux_derived'], equal_nan=True)

    def test_sync_interpolates_first_ascent_crossing_in_time_order(self):
        self.assertEqual(interpolate_speed_time([14, 10, 18, 20], [10, 0, 20, 5], 5), 12)
        self.assertEqual(interpolate_speed_time([0, 1, 2, 3], [0, np.nan, 10, 20], 5), 1)
        self.assertEqual(interpolate_speed_time([0, 1, 2], [5, 5, 10], 5), 0)
        self.assertEqual(interpolate_speed_time([0, 0, 2], [0, 10, 20], 15), 1)
        self.assertEqual(interpolate_speed_time([-2, 0, 2], [0, 10, 20], 10), 0)

    def test_sync_rejects_extrapolation_and_return_only_data(self):
        for times, speeds, target in (([0, 1], [10, 20], 5), ([0, 1], [10, 20], 30),
                                     ([0, 1], [20, 10], 15), ([0, 0], [0, 10], 5),
                                     ([0, 1], [np.nan, np.nan], 5)):
            with self.subTest(speeds=speeds, target=target), self.assertRaises(ValueError):
                interpolate_speed_time(times, speeds, target)

    def test_ecef_vectors_provide_speed_without_changing_reference_frame(self):
        data = self.load("time_s,stage1_ecef_vx_mps,stage1_ecef_vy_mps,stage1_ecef_vz_mps,stage1_speed_mps,stage1_altitude_km\n"
                         "0,3,4,0,,0\n1,3,,0,7,0\n")
        channels = data.channels('stage1', 'Stage 1 / booster')
        np.testing.assert_allclose(channels['speed'], [5, 7])
        np.testing.assert_allclose(channels['vx'], [3, 3])
        np.testing.assert_allclose(channels['pressure_derived'], .5*1.225*np.array([5, 7])**2/1000)

    def test_shipped_telemetry_formats(self):
        root = Path(__file__).resolve().parents[1] / 'falcon9_telemetry'
        paths = sorted(root.glob('*.csv'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(name=path.name):
                data = load_telemetry_csv(path)
                self.assertGreater(len(data.time), 0)
                self.assertTrue(any('speed' in group for group in data.groups.values()))
                self.assertTrue(any('altitude' in group for group in data.groups.values()))

    def test_numerical_acceleration_unequal_intervals_and_deceleration(self):
        np.testing.assert_allclose(numerical_acceleration([0, .5, 2, 4], [10, 12, 18, 14]),
                                   [np.nan, 4, 4, -2], equal_nan=True)
        np.testing.assert_allclose(numerical_acceleration([100, 100.5, 102, 104], [10, 12, 18, 14]),
                                   [np.nan, 4, 4, -2], equal_nan=True)

    def test_numerical_acceleration_gaps_duplicate_times_and_insufficient_data(self):
        np.testing.assert_allclose(numerical_acceleration([0, 1, 1, 2, 3, 4, 5],
                                                         [0, 10, 10, np.nan, 30, 40, 35]),
                                   [np.nan, 10, np.nan, np.nan, np.nan, 10, -5], equal_nan=True)
        for time, speed in (([], []), ([0], [10]), ([0, 0], [0, 10]),
                            ([1, 0], [0, 10]), ([0, 1], [-1, 10]),
                            ([0, np.nan], [0, 10]), ([0, 1], [0, np.inf])):
            with self.subTest(time=time, speed=speed):
                self.assertTrue(np.isnan(numerical_acceleration(time, speed)).all())
        with self.assertRaises(ValueError):
            numerical_acceleration([0, 1], [0])

    def test_acceleration_derived_excludes_stage1_preserves_csv_and_units(self):
        data = self.load("time_s,speed_kmh,acceleration_g,stage2_velocity_mps,booster_speed_mps\n"
                         "0,0,1,100,40\n2,72,2,140,20\n5,180,3,200,5\n")
        self.assertNotIn('acceleration_derived', data.channels('stage1', 'Stage 1 / booster'))
        for group, expected in (("stage2", [np.nan, 20, 20]),
                                ("booster", [np.nan, -10, -5])):
            channels = data.channels(group, 'Stage 1 / booster')
            np.testing.assert_allclose(channels['acceleration_derived'], expected, equal_nan=True)
        np.testing.assert_allclose(data.channels('stage1', 'Stage 1 / booster')['acceleration'],
                                   np.array([1, 2, 3]) * 9.80665)
        self.assertNotIn('acceleration_derived', data.groups['primary'])
        # Scalar speed is derived from complete ECEF vectors before differentiation.
        vector = self.load("time_s,ecef_vx_mps,ecef_vy_mps,ecef_vz_mps\n0,3,4,0\n2,6,8,0\n")
        np.testing.assert_allclose(vector.channels('stage2', 'Stage 2')['acceleration_derived'],
                                   [np.nan, 2.5], equal_nan=True)

    def test_acceleration_overlay_labels_time_shift_and_window_endpoints(self):
        data = self.load("time_s,speed_mps,acceleration_mps2\n0,0,9\n1,10,9\n2,30,9\n3,60,9\n")
        first = Figure()
        self.assertTrue(overlay_telemetry(first, 'S1 details', data, 'Stage 1 / booster', 'purple'))
        self.assertEqual([line.get_label() for line in first.axes[1].lines], ['Telemetry · CSV'])
        for phase, index, primary in (("S2 propulsion", 1, "Stage 2"),
                                      ("Booster", 4, "Booster")):
            with self.subTest(phase=phase):
                figure = Figure()
                self.assertTrue(overlay_telemetry(figure, phase, data, primary, 'purple', 5, [(6, 8)]))
                lines = {line.get_label(): line for line in figure.axes[index].lines}
                derived = lines['Telemetry · dv/dt (ECEF)']
                self.assertIn('Telemetry · CSV', lines)
                self.assertEqual(derived.get_color(), 'purple')
                np.testing.assert_allclose(derived.get_xdata(), [5, 6, 7, 8])
                np.testing.assert_allclose(derived.get_ydata(), [np.nan, np.nan, 20, 30], equal_nan=True)
                expected_title = ('Speed acceleration (ECEF, includes gravity)' if phase == 'S2 propulsion'
                                  else 'Acceleration: specific / dv/dt')
                self.assertEqual(figure.axes[index].get_title(), expected_title)
