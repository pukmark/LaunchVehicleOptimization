"""Deterministic extraction checks; no downloads, GUI, or Tesseract required."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd

import StarlinkExtract as extract


class ParsingTests(unittest.TestCase):
    def test_full_gauge_values_and_units(self):
        cases = [
            ('123.45 km', 'altitude', 'altitude', 123.45),
            ('12,345.67 km/h', 'velocity2', 'velocity', 12345.67),
            ('12 345 km/h', 'speed', 'velocity', 12345),
            ('-1.25 g', 'acceleration', 'accel_g', -1.25),
            ('0', 'altitude2', 'altitude', 0),
        ]
        for text, origin, key, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract.parse_telemetry_text(text, origin)[key], expected)
        for noise in ['123.45.6', '12 34', '123abc45', '1,23']:
            self.assertIsNone(extract.parse_telemetry_text(noise, 'speed')['velocity'])
        self.assertIsNone(extract.parse_telemetry_text('10 km/h', 'altitude')['altitude'])

    def test_timer_validation(self):
        for text, expected in [('t - 00:01:02', 'T-00:01:02'),
                               ('T+0:00:05', 'T+00:00:05'), ('00:00:00', '00:00:00')]:
            self.assertEqual(extract.parse_telemetry_text(text, 'time')['time_str'], expected)
        for text in ['T+00:99:01', 'T+12', '00:00:60', 'T+00:01:234']:
            self.assertIsNone(extract.parse_telemetry_text(text, 'time')['time_str'])

    def test_candidate_selection_uses_valid_consensus(self):
        self.assertEqual(extract._best_numeric_candidate(['123.45', '999999', '123.45', '5'], 'altitude'), '123.45')
        self.assertEqual(extract._best_numeric_candidate(['99999', 'garbage'], 'speed'), '')
        with patch.object(extract, 'ocr_telemetry', side_effect=['28', '28']) as ocr:
            self.assertEqual(extract.ocr_gauge_robust(np.zeros((10, 10, 3))), '28')
            self.assertEqual(ocr.call_count, 2)
        with patch.object(extract, 'ocr_telemetry', side_effect=['T+99:99:99', '00:00:05']):
            self.assertEqual(extract.ocr_time_robust(None), '00:00:05')

    def test_correlated_ocr_errors_do_not_outvote_otsu(self):
        with patch.object(extract, 'ocr_telemetry', side_effect=['96.8', '6.3', 'm', '6.3']):
            self.assertEqual(extract.ocr_gauge_robust(None, origin='altitude2'), '96.8')

    def test_empty_crops_and_ocr_timeouts(self):
        with self.assertRaisesRegex(ValueError, 'empty crop'):
            extract.preprocess_roi(np.zeros((0, 0, 3), dtype=np.uint8))
        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        with patch.object(extract.pytesseract, 'image_to_string', side_effect=RuntimeError('Tesseract process timeout')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(extract.ocr_telemetry(crop), '')
        with patch.object(extract.pytesseract, 'image_to_string', side_effect=RuntimeError('engine error')):
            with self.assertRaisesRegex(RuntimeError, 'engine error'):
                extract.ocr_telemetry(crop)


class CropAndFilterTests(unittest.TestCase):
    def test_crops_scale_from_4k_to_720p(self):
        crops = extract.scaled_rois((720, 1280, 3), extract.TELEMETRY_ROI_PHASE1)
        self.assertEqual(crops['speed'], {'yx1': (647, 60), 'yx2': (671, 140)})
        for shape in [(720, 1280, 3), (1080, 1920, 3), (2160, 3840, 3)]:
            for roi in extract.scaled_rois(shape, extract.TELEMETRY_ROI_PHASE2).values():
                self.assertTrue(0 <= roi['yx1'][0] < roi['yx2'][0] <= shape[0])
                self.assertTrue(0 <= roi['yx1'][1] < roi['yx2'][1] <= shape[1])
        with self.assertRaisesRegex(ValueError, 'outside'):
            extract.scaled_rois((720, 1280, 3), {'bad': {'yx1': (0, 0), 'yx2': (2200, 1)}})

    def test_filter_preserves_ramps_and_gaps_but_removes_spikes(self):
        ramp = pd.Series(np.arange(20), dtype=float)
        pd.testing.assert_series_equal(extract.filter_series_outliers(ramp), ramp)
        spike = pd.Series([10, 10, 10, 999, 10, 10, 10], dtype=float)
        result = extract.filter_series_outliers(spike)
        self.assertTrue(np.isnan(result[3]))
        self.assertEqual(result.notna().sum(), 6)
        gaps = pd.Series([None, None, 0, 1, None, 3], dtype=float)
        pd.testing.assert_series_equal(extract.filter_series_outliers(gaps), gaps)

    def test_stage2_outliers_are_not_restored(self):
        rows = [{'t_video_sec': i, 'speed_kmh': 100,
                 'velocity2_kmh': value, 'altitude2_km': value}
                for i, value in enumerate([10, 10, 10, 300, 10, 10, 10])]
        result = extract.clean_telemetry_rows(rows)
        self.assertTrue(np.isnan(result.loc[3, 'velocity2_kmh']))
        self.assertTrue(np.isnan(result.loc[3, 'altitude2_km']))
        self.assertEqual(list(result.columns), extract.CSV_COLUMNS)
        self.assertTrue(extract.clean_telemetry_rows([]).empty)


class FakeCapture:
    def __init__(self, fps=29.97, count=100):
        self.fps, self.count = fps, count
        self.position = 0
        self.retrieved = []
        self.released = False
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def get(self, key):
        return {extract.cv2.CAP_PROP_FPS: self.fps,
                extract.cv2.CAP_PROP_FRAME_COUNT: self.count}.get(key, self.position)

    def set(self, key, value):
        self.position = int(value)
        return True

    def grab(self):
        if self.position >= self.count:
            return False
        self.position += 1
        return True

    def retrieve(self):
        self.retrieved.append(self.position - 1)
        return True, self.frame

    def release(self):
        self.released = True


class ExtractionTests(unittest.TestCase):
    def run_extraction(self, cap, directory, **kwargs):
        with patch.object(extract.cv2, 'VideoCapture', return_value=cap), \
             patch.object(extract, 'ocr_time_robust', return_value='00:00:01'), \
             patch.object(extract, 'ocr_gauge_robust', return_value='1'), \
             patch.object(extract.cv2, 'imshow', side_effect=AssertionError('unexpected GUI')), \
             contextlib.redirect_stdout(io.StringIO()):
            return extract.extract_telemetry_to_csv(output_dir=directory, debug=False, **kwargs)

    def test_sampling_is_anchored_and_does_not_accumulate_rounding_drift(self):
        cap = FakeCapture(count=4000)
        with tempfile.TemporaryDirectory() as directory:
            df = self.run_extraction(cap, directory, start_time_sec=0.1, sample_fps=1, save_raw=True)
            self.assertEqual(cap.retrieved, [3 + round(i * 29.97) for i in range(len(df))])
            self.assertNotEqual(cap.retrieved[-1], 3 + 30 * (len(df) - 1))
            saved = pd.read_csv(Path(directory) / 'falcon9_telemetry_starlink.csv')
            self.assertEqual(list(saved.columns), extract.CSV_COLUMNS)
            self.assertTrue((Path(directory) / 'falcon9_telemetry_starlink_raw.csv').exists())
        self.assertTrue(cap.released)

    def test_stage_separation_and_inclusive_end(self):
        cap = FakeCapture(fps=10, count=100)
        with tempfile.TemporaryDirectory() as directory:
            df = self.run_extraction(cap, directory, start_time_sec=0.1, end_time_sec=2.1,
                                     separation_sec=1.1, sample_fps=1)
        self.assertEqual(cap.retrieved, [1, 11, 21])
        self.assertTrue(np.isnan(df.loc[0, 'velocity2_kmh']))
        self.assertEqual(df.loc[1, 'velocity2_kmh'], 1)
        self.assertTrue(np.isnan(df.loc[1, 'acceleration_g']))

    def test_invalid_settings_and_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            for settings in [{'sample_fps': 0}, {'sample_fps': float('nan')},
                             {'start_time_sec': -1}, {'start_time_sec': 3, 'end_time_sec': 2}]:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    self.run_extraction(FakeCapture(), directory, **settings)
            for fps in [0, float('nan')]:
                cap = FakeCapture(fps=fps)
                with self.assertRaisesRegex(ValueError, 'invalid FPS'):
                    self.run_extraction(cap, directory)
                self.assertTrue(cap.released)

    def test_capture_released_on_ocr_failure(self):
        cap = FakeCapture()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(extract.cv2, 'VideoCapture', return_value=cap), \
             patch.object(extract, 'ocr_gauge_robust', side_effect=RuntimeError('OCR failed')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'OCR failed'):
                extract.extract_telemetry_to_csv(output_dir=directory, debug=False)
        self.assertTrue(cap.released)

    def test_all_failed_ocr_preserves_results_and_can_save_raw(self):
        cap = FakeCapture(fps=10, count=1)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(extract.cv2, 'VideoCapture', return_value=cap), \
             patch.object(extract, 'ocr_time_robust', return_value=''), \
             patch.object(extract, 'ocr_gauge_robust', return_value='bad OCR'), \
             contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / 'falcon9_telemetry_starlink.csv'
            path.write_text('existing results')
            with self.assertRaisesRegex(RuntimeError, 'No valid telemetry'):
                extract.extract_telemetry_to_csv(output_dir=directory, debug=False, save_raw=True)
            self.assertEqual(path.read_text(), 'existing results')
            self.assertTrue((Path(directory) / 'falcon9_telemetry_starlink_raw.csv').exists())
        self.assertTrue(cap.released)

    def test_empty_segment_does_not_overwrite_results(self):
        cap = FakeCapture(fps=10, count=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'falcon9_telemetry_starlink.csv'
            path.write_text('existing results')
            with self.assertRaisesRegex(ValueError, 'No frames'):
                self.run_extraction(cap, directory, start_time_sec=0.11, end_time_sec=0.12)
            self.assertEqual(path.read_text(), 'existing results')
        self.assertTrue(cap.released)


if __name__ == '__main__':
    unittest.main()
