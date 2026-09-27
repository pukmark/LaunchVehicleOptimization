"""GUI snapshot format and interrupted-write regression checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from GUIState import read_gui_state, write_gui_state


class GUIStateTests(unittest.TestCase):
    def test_round_trip_numpy_shapes_and_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings' / 'gui_state.json'
            state = {'cases': [{}, {}, {}], 'result': {
                'x': np.arange(12.).reshape(3, 4), 'angle': np.asarray(0.5),
                'mass': np.float64(12500), 'iterations': np.int64(24)}}
            write_gui_state(path, state)
            restored = read_gui_state(path)
            np.testing.assert_array_equal(restored['result']['x'], state['result']['x'])
            self.assertEqual(restored['result']['angle'].shape, ())
            self.assertEqual(restored['result']['mass'], 12500)
            self.assertEqual(restored['result']['iterations'], 24)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_failed_replace_keeps_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gui_state.json'
            write_gui_state(path, {'cases': [{}, {}, {}], 'selected_case': 1})
            before = path.read_bytes()
            with patch('GUIState.os.replace', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    write_gui_state(path, {'cases': [{}, {}, {}], 'selected_case': 2})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_rejects_invalid_or_unknown_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gui_state.json'
            for text in ('{broken', '[]', '{"version":999,"cases":[{},{},{}]}',
                         '{"version":1,"cases":[]}'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        read_gui_state(path)
