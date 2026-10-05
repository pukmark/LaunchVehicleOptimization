"""Sanity checks and interactive corrections without video, OCR, or GUI."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

import ExtractTelemetry as extract


def sample(speed=1000, altitude=10, acceleration=1, time="T+00:00:00"):
    return {origin: extract.parse_telemetry_text(str(value), origin)
            for origin, value in (("speed", speed), ("altitude", altitude),
                                  ("acceleration", acceleration), ("time", time))}


class SanityTests(unittest.TestCase):
    def issue(self, origin, value, time, history, data=None):
        return extract._sanity_issue(origin, value, time, history, data or {})

    def test_speed_acceleration_uses_units_elapsed_time_and_strict_limit(self):
        history = {"speed": [(0, 0)]}
        threshold = 10 * 9.80665 * 3.6
        self.assertIsNone(self.issue("speed", threshold - 1, 1, history))
        self.assertIn("10.00g", self.issue("speed", threshold, 1, history))
        self.assertIsNone(self.issue("speed", threshold, 2, history))
        self.assertIsNotNone(self.issue("speed", threshold / 2, .5, history))
        self.assertIsNotNone(self.issue("velocity2", 1000, 1,
                                       {"velocity2": [(0, 2000)]}))

    def test_altitude_checks_first_change_against_vehicle_speed(self):
        history = {"altitude": [(0, 10)]}
        data = sample(speed=3600)
        self.assertIsNone(self.issue("altitude", 11, 1, history, data))
        self.assertIn("vehicle speed", self.issue("altitude", 20, 1, history, data))
        self.assertIsNone(self.issue("altitude", 12, 2, history, data))

    def test_altitude_allows_fast_steady_ascent_but_rejects_acceleration_spike(self):
        history = {"altitude": [(0, 10), (1, 11)]}
        self.assertIsNone(self.issue("altitude", 12, 2, history))
        self.assertIn("altitude trend", self.issue("altitude", 13, 2, history))
        # Three unequally spaced samples on a 5g vertical trajectory.
        altitude = lambda t: 10 + .5 * 5 * 9.80665 * t**2 / 1000
        history = {"altitude": [(0, altitude(0)), (2, altitude(2))]}
        self.assertIsNone(self.issue("altitude", altitude(5), 5, history))
        self.assertIsNotNone(self.issue("altitude", altitude(5) + 5, 5, history))

    def test_altitude_rounding_does_not_trigger_false_alarm(self):
        history = {"altitude": [(0, 10), (1, 10.1)]}
        self.assertIsNone(self.issue("altitude", 10.1, 2, history, sample(speed=400)))
        self.assertIsNotNone(self.issue("altitude", 10.5, 2, history, sample(speed=400)))

    def test_stage2_uses_its_own_history_and_speed(self):
        data = sample(speed=0)
        data["velocity2"] = {"velocity": 3600}
        history = {"altitude": [(0, 0), (1, 0)], "altitude2": [(1, 10)]}
        self.assertIsNone(self.issue("altitude2", 11, 2, history, data))
        self.assertIsNone(self.issue("velocity2", 20000, 2, {"speed": [(1, 0)]}))

    def test_acceleration_range_and_slew_limit(self):
        history = {"acceleration": [(0, -1)]}
        self.assertIsNone(self.issue("acceleration", 1, 1, history))
        self.assertIsNotNone(self.issue("acceleration", 9, 1, history))
        self.assertIsNone(self.issue("acceleration", 9, 2, history))
        self.assertIsNotNone(self.issue("acceleration", 11, 2, {}))

    def test_timer_follows_video_and_handles_countdown_crossing(self):
        history = {"time": [(0, "T-00:00:01")]}
        self.assertIsNone(self.issue("time", "T+00:00:00", 1, history))
        history = {"time": [(1, "T+00:00:00")]}
        self.assertIsNone(self.issue("time", "T+00:00:01", 2.01, history))
        self.assertIsNotNone(self.issue("time", "T+00:00:10", 2, history))
        self.assertIsNotNone(self.issue("time", "T-00:00:01", 2, history))

    def test_missing_nonfinite_and_out_of_range_values_are_rejected(self):
        for origin in extract.SANITY_FIELDS:
            self.assertIsNotNone(self.issue(origin, None, 0, {}))
        for origin, value in (("speed", -1), ("speed", 30001),
                              ("altitude", 401), ("altitude2", -1),
                              ("acceleration", np.inf), ("speed", np.nan)):
            with self.subTest(origin=origin, value=value):
                self.assertIsNotNone(self.issue(origin, value, 0, {}))

    def test_manual_correction_retries_bad_input_and_updates_reference(self):
        history = {"speed": [(0, 1000)]}
        data = {"speed": extract.parse_telemetry_text("5000", "speed")}
        with patch("builtins.input", side_effect=["bad", "nan", "inf", "30001", "1050 km/h"]) as prompt, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            extract.validate_telemetry_sample(data, {"speed": "5000"}, 1, 30, history)
            self.assertEqual(prompt.call_count, 5)
            data = {"speed": extract.parse_telemetry_text("1100", "speed")}
            extract.validate_telemetry_sample(data, {"speed": "1100"}, 2, 60, history)
            self.assertEqual(prompt.call_count, 5)
        self.assertEqual(history["speed"], [(1, 1050), (2, 1100)])
        self.assertIn("frame 30", output.getvalue())
        self.assertIn("Last valid: 1000", output.getvalue())

    def test_manual_confirmation_can_override_continuity_warning(self):
        data = {"speed": extract.parse_telemetry_text("5000", "speed")}
        history = {"speed": [(0, 1000)]}
        with patch("builtins.input", return_value="5000"), contextlib.redirect_stdout(io.StringIO()):
            extract.validate_telemetry_sample(data, {}, 1, 1, history)
        self.assertEqual(history["speed"][-1], (1, 5000))

    def test_each_failed_active_value_gets_its_own_prompt(self):
        data = sample(speed="bad", altitude="bad", acceleration="bad", time="bad")
        history = {}
        with patch("builtins.input", side_effect=["100", "0.1", "1.5", "bad", "T-00:00:02"]) as prompt, \
             contextlib.redirect_stdout(io.StringIO()):
            extract.validate_telemetry_sample(data, {}, 0, 0, history)
        self.assertEqual(prompt.call_count, 5)
        self.assertEqual(history["time"], [(0, "T-00:00:02")])
        self.assertNotIn("altitude2", history)
        self.assertNotIn("velocity2", history)


class RecoveryModeTests(unittest.TestCase):
    def test_exp_disables_booster_gauges_exactly_at_separation(self):
        shape = (720, 1280, 3)
        before = extract.rois_at_time(shape, .9, 1, "EXP")
        after = extract.rois_at_time(shape, 1, 1, "EXP")
        self.assertEqual(set(before), {"speed", "altitude", "acceleration", "time"})
        self.assertEqual(set(after), {"velocity2", "altitude2", "time"})
        self.assertEqual(after["velocity2"], extract.rois_at_time(shape, 1, 1, "ASDS")["velocity2"])
        self.assertEqual(extract.rois_at_time(shape, 2, None, "EXP"), before)

    def test_recovery_modes_keep_booster_gauges(self):
        for mode in ("ASDS", "RTLS"):
            with self.subTest(mode=mode):
                rois = extract.rois_at_time((720, 1280, 3), 1, 1, mode)
                self.assertEqual(set(rois), {"speed", "altitude", "velocity2", "altitude2", "time"})

    def test_mode_validation_and_configured_default(self):
        self.assertEqual(extract.normalize_recovery_mode(" exp "), "EXP")
        with patch.object(extract, "RECOVERY_MODE", "EXP"):
            self.assertEqual(set(extract.rois_at_time((720, 1280, 3), 1, 1)),
                             {"velocity2", "altitude2", "time"})
        with patch.object(extract.cv2, "VideoCapture") as capture:
            for mode in ("unknown", "", 1):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "recovery mode"):
                    extract.extract_telemetry_to_csv(recovery_mode=mode)
            capture.assert_not_called()

    def test_cli_passes_mode_to_extraction_and_preview(self):
        with patch.object(extract, "extract_telemetry_to_csv") as run:
            extract.main(["--recovery-mode", "exp"])
            self.assertEqual(run.call_args.kwargs["recovery_mode"], "EXP")
        with patch.object(extract, "show_sample_frame") as preview, \
             patch.object(extract, "extract_telemetry_to_csv") as run:
            extract.main(["--recovery", "EXP", "--preview"])
            self.assertEqual(preview.call_args.kwargs["recovery_mode"], "EXP")
            run.assert_not_called()


class FakeCapture:
    def __init__(self, count):
        self.count, self.position = count, 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return {extract.cv2.CAP_PROP_FPS: 1,
                extract.cv2.CAP_PROP_FRAME_COUNT: self.count}[key]

    def grab(self):
        self.position += 1
        return self.position <= self.count

    def retrieve(self):
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class WindowTestCase(unittest.TestCase):
    def setUp(self):
        self.gui = {}
        for name in ("namedWindow", "imshow", "setWindowProperty", "waitKey", "destroyWindow"):
            mock = patch.object(extract.cv2, name, return_value=-1 if name == "waitKey" else None)
            self.gui[name] = mock.start()
            self.addCleanup(mock.stop)


class CorrectionPreviewTests(WindowTestCase):
    def test_shows_matching_crop_before_each_prompt_and_keeps_it_for_retries(self):
        data = {origin: extract.parse_telemetry_text("bad", origin)
                for origin in ("speed", "altitude")}
        crops = {"speed": np.full((24, 80, 3), 50, dtype=np.uint8),
                 "altitude": np.full((24, 80, 3), 100, dtype=np.uint8)}
        responses = iter([("speed", "bad"), ("speed", "100"), ("altitude", "0.1")])

        def enter_value(prompt):
            origin, response = next(responses)
            self.assertIn(origin, prompt)
            window, preview = self.gui["imshow"].call_args.args
            self.assertEqual(window, f"Manual correction - {origin}")
            self.assertTrue((preview[60:280, 10:743] == crops[origin][0, 0]).all())
            self.assertEqual(self.gui["waitKey"].call_args.args, (100,))
            self.assertEqual(self.gui["destroyWindow"].call_count, 0 if origin == "speed" else 1)
            return response

        with patch("builtins.input", side_effect=enter_value), contextlib.redirect_stdout(io.StringIO()):
            extract.validate_telemetry_sample(data, {"speed": "bad", "altitude": "bad"},
                                              12, 360, {}, roi_images=crops)
        self.assertEqual(self.gui["imshow"].call_count, 2)
        self.assertEqual([call.args[0] for call in self.gui["destroyWindow"].call_args_list],
                         ["Manual correction - speed", "Manual correction - altitude"])

    def test_valid_values_do_not_open_correction_window(self):
        with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
            extract.validate_telemetry_sample(sample(), {}, 0, 0, {},
                                              roi_images={"speed": np.zeros((24, 80, 3), dtype=np.uint8)})
        self.gui["imshow"].assert_not_called()

    def test_window_closes_when_input_is_interrupted(self):
        data = {"time": extract.parse_telemetry_text("bad", "time")}
        with patch("builtins.input", side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            extract.validate_telemetry_sample(data, {}, 0, 0, {},
                                              roi_images={"time": np.zeros((24, 80, 3), dtype=np.uint8)})
        self.gui["destroyWindow"].assert_called_once_with("Manual correction - time")


class ExtractionTests(WindowTestCase):
    def test_exp_skips_booster_ocr_but_still_checks_stage2_values(self):
        cap = FakeCapture(3)

        def gauge(_image, origin):
            if cap.position > 1 and origin in ("speed", "altitude", "acceleration"):
                self.fail("EXP attempted to read absent booster telemetry")
            if origin == "velocity2":
                return "bad" if cap.position == 2 else "20000"
            return "100" if origin == "altitude2" else "1"

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(extract, "VIDEO_FILENAME", Path(directory) / "test.mp4"), \
             patch.object(extract.cv2, "VideoCapture", return_value=cap), \
             patch.object(extract, "ocr_gauge_robust", side_effect=gauge), \
             patch.object(extract, "ocr_time_robust", side_effect=[f"00:00:0{i}" for i in range(3)]), \
             patch("builtins.input", return_value="20000") as prompt, \
             contextlib.redirect_stdout(io.StringIO()):
            result = extract.extract_telemetry_to_csv(output_dir=directory, debug=False,
                                                       separation_sec=1, recovery_mode="EXP")
            prompt.assert_called_once()
            self.assertIn("velocity2", prompt.call_args.args[0])
            self.assertEqual(self.gui["imshow"].call_args.args[0], "Manual correction - velocity2")
            self.assertEqual(result.loc[0, "speed_kmh"], 1)
            self.assertEqual(result.loc[0, "altitude_km"], 1)
            self.assertTrue(result.loc[1:, ["speed_kmh", "altitude_km", "acceleration_g"]].isna().all().all())
            self.assertEqual(result.loc[1:, "velocity2_kmh"].tolist(), [20000, 20000])
            self.assertEqual(result.loc[1:, "altitude2_km"].tolist(), [100, 100])
            csv_path = next(path for path in Path(directory).glob("*.csv")
                            if not path.stem.endswith("_raw"))
            saved = pd.read_csv(csv_path)
            self.assertEqual(list(saved.columns), extract.CSV_COLUMNS)
            self.assertTrue(saved.loc[1:, "speed_kmh"].isna().all())
        self.assertTrue(cap.released)

    def test_pauses_resumes_and_saves_corrected_values_without_median_filter(self):
        cap = FakeCapture(7)
        speeds = iter([10, 10, 10, 1500, 10, 10, 10])

        def gauge(_image, origin):
            return str(next(speeds)) if origin == "speed" else "1"

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(extract, "VIDEO_FILENAME", Path(directory) / "test.mp4"), \
             patch.object(extract.cv2, "VideoCapture", return_value=cap), \
             patch.object(extract, "ocr_gauge_robust", side_effect=gauge), \
             patch.object(extract, "ocr_time_robust", side_effect=[f"00:00:0{i}" for i in range(7)]), \
             patch("builtins.input", side_effect=["999", "10"]) as prompt, \
             contextlib.redirect_stdout(io.StringIO()):
            result = extract.extract_telemetry_to_csv(output_dir=directory, debug=False,
                                                       separation_sec=None, save_raw=True)
            self.assertEqual(prompt.call_count, 2)
            self.assertEqual(len(result), 7)
            self.assertEqual(result.loc[3, "speed_kmh"], 999)
            csv_path = next(path for path in Path(directory).glob("*.csv")
                            if not path.stem.endswith("_raw"))
            saved = pd.read_csv(csv_path)
            self.assertEqual(saved.loc[3, "speed_kmh"], 999)
            raw = pd.read_csv(next(Path(directory).glob("*_raw.csv")))
            self.assertEqual(raw.loc[3, "raw_speed"], 1500)
            self.assertEqual(raw.loc[3, "speed_kmh"], 999)
        self.assertTrue(cap.released)

    def test_stage_separation_checks_new_gauges_without_prompting_for_absent_ones(self):
        cap = FakeCapture(3)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(extract, "VIDEO_FILENAME", Path(directory) / "test.mp4"), \
             patch.object(extract.cv2, "VideoCapture", return_value=cap), \
             patch.object(extract, "ocr_gauge_robust", return_value="1"), \
             patch.object(extract, "ocr_time_robust", side_effect=[f"00:00:0{i}" for i in range(3)]), \
             patch("builtins.input", side_effect=AssertionError("unexpected prompt")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = extract.extract_telemetry_to_csv(output_dir=directory, debug=False,
                                                       separation_sec=1)
        self.assertTrue(np.isnan(result.loc[0, "velocity2_kmh"]))
        self.assertEqual(result.loc[1, "velocity2_kmh"], 1)
        self.assertTrue(np.isnan(result.loc[1, "acceleration_g"]))
        self.assertTrue(cap.released)

    def test_missing_terminal_input_releases_capture_without_overwriting_csv(self):
        cap = FakeCapture(1)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(extract, "VIDEO_FILENAME", Path(directory) / "test.mp4"), \
             patch.object(extract.cv2, "VideoCapture", return_value=cap), \
             patch.object(extract, "ocr_gauge_robust", return_value="bad"), \
             patch.object(extract, "ocr_time_robust", return_value="00:00:00"), \
             patch("builtins.input", side_effect=EOFError), \
             contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "test.csv"
            path.write_text("existing results")
            with self.assertRaisesRegex(RuntimeError, "interactive terminal"):
                extract.extract_telemetry_to_csv(output_dir=directory, debug=False)
            self.assertEqual(path.read_text(), "existing results")
        self.assertTrue(cap.released)
        self.gui["destroyWindow"].assert_called_once_with("Manual correction - speed")


if __name__ == "__main__":
    unittest.main()
