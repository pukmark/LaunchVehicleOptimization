"""Versioned, atomic GUI snapshots using JSON (no executable deserialization)."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


STATE_VERSION = 1


def _encode(value):
    if isinstance(value, np.ndarray):
        return {"__gui_array__": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot save {type(value).__name__}")


def _decode(value):
    if set(value) == {"__gui_array__"}:
        return np.asarray(value["__gui_array__"], dtype=float)
    return value


def read_gui_state(path):
    state = json.loads(Path(path).read_text(encoding="utf-8"), object_hook=_decode)
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        raise ValueError("Unrecognized GUI settings format")
    if not isinstance(state.get("cases"), list) or len(state["cases"]) != 3:
        raise ValueError("GUI settings must contain three cases")
    return state


def write_gui_state(path, state):
    """Replace the old snapshot only after its replacement is fully written."""
    path = Path(path)
    payload = json.dumps(dict(state, version=STATE_VERSION), default=_encode,
                         ensure_ascii=False, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".gui_state_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
