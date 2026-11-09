"""GUI worker files are temporary while session results remain restorable."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from GUIState import read_gui_state, write_gui_state
from OptimizationGUI import OptimizationApp


class WorkerStorageTests(unittest.TestCase):
    def test_optimization_fit_and_sensitivity_jobs_do_not_create_saved_case_folders(self):
        app = OptimizationApp.__new__(OptimizationApp)
        with tempfile.TemporaryDirectory() as saved:
            app.output_root = Path(saved)
            self.addCleanup(app._cleanup_worker_workspace)
            directories = []
            for prefix in ('case1_', 'fit_case2_', 'sensitivity_case3_'):
                directory = app._new_job_directory(prefix)
                self.assertTrue(directory.is_dir())
                self.assertFalse(directory.is_relative_to(app.output_root))
                for previous in directories:
                    self.assertFalse(previous.exists())
                directories.append(directory)
                (directory / 'solver.log').write_text('worker output')
            self.assertEqual(len(set(directories)), 3)
            self.assertEqual(list(app.output_root.iterdir()), [])
            app._cleanup_worker_workspace()
            self.assertTrue(all(not directory.exists() for directory in directories))
            app._cleanup_worker_workspace()  # Closing twice is harmless.

    def test_explicit_session_save_survives_temporary_worker_cleanup(self):
        app = OptimizationApp.__new__(OptimizationApp)
        self.addCleanup(app._cleanup_worker_workspace)
        worker = app._new_job_directory('case1_')
        result = {'success': True, 'x1': np.array([[1., 2.], [3., 4.]])}
        with tempfile.TemporaryDirectory() as saved:
            path = Path(saved) / 'gui_state.json'
            cases = [{'result': result, 'run_dir': None, 'request': {'configuration': 1}},
                     {'fit_report': {'termination': 'Complete'}},
                     {'sensitivity_report': {'termination': 'Complete'}}]
            write_gui_state(path, {'cases': cases})
            app._cleanup_worker_workspace()
            self.assertFalse(worker.exists())
            restored = read_gui_state(path)
            np.testing.assert_array_equal(restored['cases'][0]['result']['x1'], result['x1'])
            self.assertIsNone(restored['cases'][0]['run_dir'])
            self.assertEqual(restored['cases'][1]['fit_report']['termination'], 'Complete')
            self.assertEqual(restored['cases'][2]['sensitivity_report']['termination'], 'Complete')
