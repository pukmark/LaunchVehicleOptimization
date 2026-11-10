# Launch Vehicle Optimization

A Python research environment for optimizing satellite launch trajectories and
booster recovery using CasADi and IPOPT. It models first-stage ascent, upper-stage
flight, fairing separation, final orbit adjustment, and optional booster return.
You can maximize payload, solve for a fixed payload, compare trajectories with
flight telemetry, fit model parameters, and explore payload sensitivity.

The desktop GUI supports three comparison cases, custom vehicle definitions,
launch-site selection, flight-phase plots, and a 3D trajectory view. Built-in
vehicle models include Falcon 9, Falcon 9 True, Starship, Starship V3, and New Glenn. Their
parameters include engineering estimates; solver convergence and model fidelity
depend on the selected case.

## Prerequisites

- **Python 3.10**, the version used by the pinned environment. Newer Python
  versions have not been validated with these pins.
- **Git** to clone the repository, or download and extract its ZIP archive.
- **Tk bindings and a desktop display** for the GUI and interactive plots.
- **FFmpeg and Tesseract OCR** only if extracting telemetry from videos.

On Ubuntu 22.04 with Python 3.10, install the system packages with:

```bash
sudo apt update
sudo apt install python3.10-venv python3-tk
# Optional: video downloads and telemetry OCR
sudo apt install ffmpeg tesseract-ocr
```

On other Linux distributions, install the equivalent packages for your Python
interpreter. On macOS or Windows, install Python 3.10 with working Tk support;
if using video extraction, make sure `ffmpeg` and `tesseract` are on `PATH`.
The pinned environment was verified on Linux; other platforms are unverified.

CasADi provides the IPOPT interface and MUMPS linear solver used by the current
optimizer. No standalone `ipopt` executable or HSL installation is required.

## Installation

Clone the repository, or download and extract its ZIP archive. For a fork, use
your fork's URL instead. Skip the clone step if you downloaded a ZIP, and open
a terminal in its extracted project folder.

```bash
git clone https://github.com/pukmark/LaunchVehicleOptimization.git
cd LaunchVehicleOptimization
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

For Windows PowerShell, create and activate the environment with these commands,
then run the same `python -m pip ...` commands above:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Select `.venv/bin/python` (Windows: `.venv/Scripts/python.exe`) as your editor's
interpreter. Run the commands below from the repository root with this
environment activated. Each new terminal needs the activation command again.

`requirements.txt` pins the direct dependencies for optimization, plotting,
CSV comparison, and Tesseract-based video extraction. The optional
`requirements-ocr.txt` also installs PaddleOCR and PaddlePaddle for the secondary
OCR reader:

```bash
python -m pip install -r requirements-ocr.txt
```

The secondary reader downloads its model weights on first use and caches them
under `.cache/paddleocr/`. It is unnecessary for the GUI, optimization, tests,
or loading the included telemetry CSV files. Dependency files pin direct
packages; transitive dependencies are resolved by pip during installation.

## First run: desktop GUI

```bash
python OptimizationGUI.py
```

1. Start with **Case 1** and the default Falcon 9, expendable recovery (`EXP`),
   and 200 × 200 km orbit at 28.6° inclination.
2. Leave **Payload mass [kg]** at `-1` to maximize payload, or enter a positive
   value to fix the payload mass.
3. Click **Run Case 1**. Follow progress in **Solver output**; **Cancel** stops
   the active worker. Repeated runs reuse the selected case’s last successful
   trajectory for nearby dispersion changes with the same vehicle, recovery,
   launch site, orbit, and payload setting. Incompatible or incomplete saved
   trajectories fall back to a generated initial guess.
4. After convergence, inspect the first-stage, upper-stage, and 3D trajectory
   tabs. **Summery**, the leftmost tab, combines altitude, Earth-fixed speed, mass,
   throttle, and AoA/thrust angle on a shared timeline, including coasts.
   RTLS booster histories include the post-separation coast before boostback,
   using `SecondStage_CoastTimeAfterSep`. Throttle is zero during this coast,
   and boostback angle curves begin at ignition. Rerun older cases to regenerate
   their histories with the added phase.
   Legends contain only **Main vehicle** (phases 1–3) and **Booster**, and are
   omitted when only one group is plotted.
   Ascent uses solid lines and booster return uses dashed lines. Select another
   case to compare a different vehicle, orbit, or recovery strategy.
   Loaded telemetry overlays any available altitude, speed, mass, throttle, and
   angle measurements in this tab, using the telemetry color, time shift, and
   column routing. It can also display telemetry without a computed case.
   **Booster 2** plots altitude vertically against speed, mass, dynamic pressure,
   heat flux, and acceleration, with available booster telemetry overlays.
   Use **Import settings** beside a case to copy another case's last-run inputs
   (or its current inputs if it has not been run), then run the destination case.
5. Use **Load telemetry CSV…** to overlay a file from `falcon9_telemetry/`, then
   **Sync to Case N** to align it with a computed case.

Recovery choices are `EXP` (expendable), `ASDS` (drone-ship landing), and `RTLS`
(return to launch site). Orbit fields in the GUI use altitude in kilometres
and inclination in degrees. Recovery cases add booster-return plots.

GUI runs, fits, and sensitivities use temporary worker files, removed before the
next job or when the GUI closes; they do not create per-run folders in `Results/gui/`. The GUI automatically
restores its session from `Results/gui/gui_state.json`; **Save state…** and
**Load state…** transfer a complete session. **Reset GUI** restores defaults.
These local outputs are excluded from Git.

## Running without the GUI

The solver can also be called from Python without a display. For example, save
the following as `run_case.py` in the repository root and run `python run_case.py`:

```python
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import asdict
from pathlib import Path
import pickle
from LV_Type_scaled import DispesrionFactorsType
from OptimizationGUI import nominal_request, run_optimization

request = nominal_request(
    asdict(DispesrionFactorsType()),
    configuration=1,       # 1: Falcon 9, 2: Starship, 3: Starship V3, 4: New Glenn, 5: Falcon 9 True
    recovery="EXP",        # EXP, ASDS, or RTLS
    payload_mass=-1,        # kg; non-positive values maximize payload
    orbit_values={"apogee": "200", "perigee": "200", "inclination": "28.6"},
)
solution = run_optimization(request, output_dir=None)
print("Converged:", solution.get("success"))
if solution.get("success"):
    print("Payload [kg]:", solution["payload_mass"])
output = Path("Results/example")
output.mkdir(parents=True, exist_ok=True)
with (output / "solution.pickle").open("wb") as stream:
    pickle.dump(solution, stream)
```

This wrapper accepts the same orbit units as the GUI. The lower-level
`LV_Optimization.SolveOptimiztion` API uses geocentric orbit radii in metres and
inclination in radians. Keep that distinction when constructing requests directly.
Standalone automatic result saving is currently disabled in the optimizer, so
this example explicitly saves the returned dictionary. The GUI worker saves
its own results independently.

`python Main.py` runs the existing batch/sensitivity experiment. Edit its
`__main__` configuration before running it: the checked-in experiment selects
New Glenn with ASDS recovery, unlike the GUI defaults.

## Telemetry extraction

The included CSV files can be loaded immediately; source videos are excluded
from Git. To extract your own video, first install FFmpeg and Tesseract, then
edit the broadcast settings and crop rectangles in `ExtractTelemetry.py`.
Use `--preview` to inspect the crops:

```bash
python ExtractTelemetry.py --help
python ExtractTelemetry.py --video path/to/broadcast.mp4 --preview
python ExtractTelemetry.py --video path/to/broadcast.mp4 --save-raw
```

Set the start, end, separation, and landing timestamps for that broadcast.
Without `--video`, the script uses or downloads its configured broadcast with
yt-dlp. Keep downloaded `.mp4.clip.json` metadata next to its video so original
broadcast timestamps remain correct. Unreadable values can trigger manual
correction prompts even when plot windows are disabled.

See the [detailed usage guide](docs/USAGE.md) for CSV columns, reference frames,
telemetry fitting, sensitivities, custom vehicles, saved states, and OCR settings.

## Project layout

| File or directory | Purpose |
| --- | --- |
| `OptimizationGUI.py` | Desktop interface and solver worker |
| `LV_Optimization_Type.py` | Multiphase optimal-control problem and solver |
| `LV_Type_scaled.py`, `Atmosphere_Type.py` | Vehicle dynamics and atmosphere |
| `VehicleDefinitions.py`, `LaunchSites.py` | Built-in/custom vehicles and launch sites |
| `Main.py` | Scripted experiments, sweeps, and plots |
| `DispersionFit.py`, `PayloadSensitivity.py` | Telemetry fitting and payload sensitivity |
| `TelemetryCSV.py`, `TelemetryFrames.py`, `Trajectory3D.py` | Telemetry loading, frame conversions, and 3D plots |
| `ExtractTelemetry.py`, `StarlinkExtract.py` | Broadcast telemetry extraction |
| `falcon9_telemetry/` | Included sample telemetry CSV files |
| `tests/` | Existing unittest regression suite |
| `docs/USAGE.md` | Detailed feature and model documentation |
| `Papers/` | Reference documents; unnecessary for running the code |
| `Results/` | Generated runs and GUI state; created at runtime and ignored by Git |

Other `Main_*`, `ReturnPhase*`, and unscaled-model scripts are historical or
experimental. Some hard-code an HSL path or require a generated
`ReturnSolution.pickle`; use the current GUI or API for the setup described here.
The legacy return scripts generate that pickle locally.

## Verification and troubleshooting

Run the existing checks after installing `requirements.txt`:

```bash
python -m pip check
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

For PowerShell, set `$env:MPLBACKEND = "Agg"` before the unittest command.
The tests use Python's standard `unittest` module. They exercise dynamics,
serialization, telemetry, GUI logic, and solver setup without requiring a full
converged optimization; display-dependent checks skip when no display is available.

The current suite has two known failing checks: standalone result-file saving
(currently disabled in the optimizer) and a 3D GUI test fixture missing sensitivity
panel state. See [known verification limits](docs/USAGE.md#known-verification-limits).

- **No module named `tkinter`:** install Tk bindings for the Python interpreter
  used to create `.venv`. `python -m tkinter` opens a small window to check them.
- **No display / `$DISPLAY` error:** use a desktop session for the GUI, or use
  the API example with the `Agg` backend for headless work.
- **MA97 or HSL loading error:** use `LV_IPOPT_LINEAR_SOLVER=mumps python OptimizationGUI.py`
  on Linux/macOS. The optimizer otherwise selects MA97 if
  `/usr/local/lib/libcoinhsl.so` exists; `LV_IPOPT_HSL_LIBRARY` overrides its path.
- **ROS or another environment overrides packages:** on Linux/macOS, launch
  with `PYTHONPATH= python OptimizationGUI.py` to exclude external Python paths.
- **OCR executable missing:** check `ffmpeg -version` and `tesseract --version`
  and make sure both executables are on `PATH`.
- **Optimization fails:** inspect the **Solver output** tab and the current case inputs. A successful installation does not guarantee convergence for every
  vehicle, orbit, or recovery configuration.

## Preparing a Git upload

The repository ignores virtual environments, caches, solver outputs, local
credentials, downloaded videos, and the local IPOPT executable. Keep source
code, dependency files, documentation, and sample CSV files under version control.
Review the changes and commit them from your local checkout:

```bash
git status --short
git diff --check
git diff
git add README.md docs/USAGE.md requirements.txt requirements-ocr.txt .gitignore .gitattributes
git diff --cached --stat
git commit -m "Document installation and usage; clean repository artifacts"
git push origin main
```

Generated files remain local when removed from Git tracking. Ignore rules do
not rewrite earlier commits, so files committed previously remain in Git history.

Telemetry fitting supports **Dispersion factors** and **Physical vehicle parameters**
under **Fit model to telemetry…**. Physical mode exposes all custom-vehicle fields,
including independent first-stage sea-level/vacuum thrust and Isp, stage masses,
aerodynamics, and ascent/recovery dynamic-pressure limits. Select fields and edit
bounds in the displayed units. The current vehicle dispersion is baked into the
starting values; atmospheric density and launch-altitude offsets remain fixed.
The destination receives a reusable custom vehicle with neutral vehicle dispersion.
As with dispersion fits, payload stays fixed and each candidate is synchronized.
Only parameters that affect the chosen trajectory can be identified from its telemetry.

Python callers can use `fit_dispersion_to_telemetry(..., fit_mode='vehicle')`.
`parameters` selects attribute names from `VehicleDefinitions.VEHICLE_PARAMETERS`;
omitting it selects all fields. API bounds use SI units, and
`default_fit_bounds(request, fit_mode='vehicle')` supplies initial ranges.

**Falcon 9 True** is built-in vehicle configuration `5`, defined by
`VLType.Falcon9TrueDatabase()` in `LV_Type_scaled.py`. It uses the parameters
promoted from the GUI's custom library. Older sessions with identical definitions
use the built-in option; differing historical definitions remain custom variants.

`Main.py` prints the vehicle actually selected by each sweep and uses deterministic
initial estimates. Fixed-payload failures are not retried with ineffective random
payload estimates. Successful starts are cached in `Results/initial_guesses/` for
repeat runs; changes to the model, scenario, dispersion, or solver source invalidate
the corresponding cache. Failed sweep points retain the preceding successful
trajectory for subsequent points. Set `InitialGuessCache = None` to disable caching.

Compare the three assessment missions with:

```bash
.venv/bin/python CompareMultipleTelemetry.py
```

This loads the local SXM-11, CSG-3, and MRV–MEP CSVs, optimizes each using
Falcon 9 True, and saves `CompareMultipleTelemetry.png`: ECEF speed above
altitude, with one column per mission. Use `--no-show` for batch runs and
`--output path.png` to choose the figure destination. Blue curves are simulation;
orange points are telemetry. No per-mission result directories are created.

SXM-11 and CSG-3 use the supplied fixed payloads (6,950 and 2,205 kg).
MRV–MEP defaults to an explicitly labelled optimized payload estimate; specify
`--mrv-payload MASS_KG` when a fixed combined payload is available. Coordinates
use existing regional launch-site presets. Mission times are preserved; optional
`--time-shifts SXM11 CSG3 MRV` adds explicit offsets to telemetry timestamps.
CSG-3 uses direct-injection terminal constraints for its single upper-stage burn
(`Target_Orbit['direct_injection'] = True`). GTO missions retain the existing
parking-orbit and later-burn propellant-reserve model: later coast/restart is not
propagated in these curves. Defaults elsewhere retain the parking-orbit model.
If a mission fails to converge, its panels show telemetry only with a failure
annotation; the script reports the solver status and exits with a nonzero code.
The iteration and per-attempt IPOPT CPU limits are configurable with `--max-iter`
and `--max-cpu-time`. The initial validation converged for SXM-11 and MRV–MEP;
CSG-3 direct injection reached the iteration limit with the current model.
