# LaunchVehicleOptimization
Trajectory Optimization of a Satellite launch vehicle with booster return

Use the project environment (Python 3.10):

```bash
source .venv/bin/activate
python Main.py
```

To recreate it:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In your editor, select `.venv/bin/python` as the Python interpreter. The
requirements include the simulation and telemetry tools. Video downloads and
OCR additionally use the system `ffmpeg` and `tesseract` executables; interactive
Tk plots use the system Python Tk bindings. These are already installed on the
current machine. CasADi includes IPOPT and MUMPS; MA97 uses the optional system
HSL library described below.

If your shell loads ROS through `PYTHONPATH`, prefix commands with `PYTHONPATH=`
to exclude those external packages, for example `PYTHONPATH= python Main.py`
or `PYTHONPATH= python -m pip check`.

Run regression checks with `MPLBACKEND=Agg python3 -m unittest discover -s tests -v`.
The checks require NumPy, SciPy, CasADi, and Matplotlib. They exercise dynamics,
sensitivity sweeps, and result/warm-start round trips without requiring IPOPT
convergence.

`LV_Optimization.SolveOptimiztion` accepts `ipopt_options` to override IPOPT
settings, for example `{'linear_solver': 'mumps', 'max_iter': 250}`. The default
uses MA97 when `/usr/local/lib/libcoinhsl.so` exists, otherwise MUMPS.
`LV_IPOPT_LINEAR_SOLVER` and `LV_IPOPT_HSL_LIBRARY` can override those defaults.
Plotting respects Matplotlib's configured backend, including `MPLBACKEND=Agg`.

Results are written to `output_dir` (default `Results`; use `None` to disable
saving). Filenames include vehicle configuration and a hash of the precise
orbit, payload setting, and dispersion values so different cases do not
overwrite each other. Each result contains the configuration and dispersion
metadata, diagnostics, and solver status. Repeating the same case replaces its
previous result; failed solves use a separate `_Failed` filename.

New results are marked `solution_units='SI'`: states, thrusts, and time steps
are stored in physical units. Warm starts also accept older unmarked result
dictionaries, converting their scaled `dt4_landing` and `u4_reentry` fields.
Legacy reentry controls are interpreted using the current model's thrust scale;
use matching vehicle/engine settings when importing an old result.
Sensitivity sweeps accept every `DispesrionFactorsType` field, plus the legacy
`StagePartition` alias for `StagePartitionDelta`; unknown names raise an error.

Starlink telemetry extraction can run without GUI windows and accepts video
times in seconds. For a short check of the local broadcast:

```bash
PYTHONPATH= MPLBACKEND=Agg .venv/bin/python StarlinkExtract.py \
  --video falcon9_telemetry/falcon9_starlink_2026.mp4 \
  --start 610 --end 620 --output-dir /tmp/starlink-check --no-plot --save-raw
```

Use `--preview` to inspect crop rectangles, `--debug` to display OCR while
extracting, `--sample-fps` to adjust sampling, and `--separation` to set the
video timestamp where stage 2 telemetry appears. Crops automatically scale
from `ROI_REFERENCE_SIZE` (3840x2160) to the input video size; a different
broadcast layout still requires tuning the ROI settings. `--save-raw` writes
a separate CSV containing selected OCR text and unfiltered readings. The
main telemetry CSV keeps its existing seven columns for simulation comparison.
With no `--video`, the script uses or downloads the configured broadcast.
