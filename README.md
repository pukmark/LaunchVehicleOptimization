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

Launch the optimization GUI with:

```bash
PYTHONPATH= .venv/bin/python OptimizationGUI.py
```

The GUI starts with nominal `DispesrionFactorsType` defaults, Falcon 9,
expendable flight, and a 200 × 200 km LEO target at 28.6° inclination, maximizing
payload. The **Target orbit** fields let you edit apogee and perigee altitudes
in km above Earth’s reference radius, and inclination in degrees (0–180).
Apogee must be at least perigee, and both altitudes must be positive. Edit these
or any dispersion value in the right panel, then click **Run Case 1** at
the bottom right. The selected orbit is saved with the run and used in the
result plots. Multipliers are marked `×`; mass and
altitude offsets use kg and m, and stage partition is a fractional offset.
**Reset dispersion to defaults** restores the nominal factors.

**Launch site** is saved independently for each case. Presets include Kennedy
Space Center (the original nominal location), Vandenberg, Starbase (Boca Chica),
Kourou, Equator, and North Pole. Select **Custom** or edit any coordinate to enter
latitude (degrees north), longitude (degrees east; west is negative), and altitude
(metres). Editing a coordinate switches the preset to Custom. Effective launch
altitude is the site altitude plus `LaunchAltDelta`. Coordinates feed all flight
models and the local/ECEF telemetry comparisons; computed curves retain their
original site until that case is recomputed. Target orbit inclination is edited
separately, including 0° equatorial and 90° polar targets.

Presets are approximate regional coordinates with zero default altitude, not
surveyed pad elevations. Kennedy retains 28.6° N, 80.6° W for compatibility;
Vandenberg and Kourou follow [NASA/JPL launch-site references, table 6-1](https://descanso.jpl.nasa.gov/monograph/series12/LunarTraj--Overall.pdf).
Starbase uses the [USACE Boca Chica facility location](https://www.swg.usace.army.mil/Portals/26/docs/regulatory/PN%20APRIL/PN2_201200381.pdf?ver=mYDsA57a8HMJK0MxAGcVxg%3D%3D).
Equator (0°, 0°) and North Pole (90°, 0°) are idealized reference sites; longitude
at the pole defines the local heading convention. Site selection supplies launch
geometry; geographic flight corridors are not modeled. Old saved cases load at
Kennedy; **Reset GUI** also restores Kennedy for all three cases.

**Payload mass [kg]** is saved independently for each case. Enter a positive
value to fix the payload mass, or zero/any negative value to maximize payload
(the default is `-1`). The computed case summary indicates whether its payload
was fixed or maximized.

Use the three case slots above the plots to compare runs. Select a case to
edit its own vehicle, recovery, orbit, and dispersion settings; switching cases
preserves even unfinished edits. Click **Run Case N** or **Recompute Case N**
to solve the selected case. Each case has a distinct color (click **Color** to
change it) and a **Display** checkbox. All displayed, computed cases share the
same axes and legends in each flight-phase tab. Booster plots include only
cases with recovery results.

The GUI automatically saves all three case setups (including unfinished edits),
latest completed results, case colors and visibility, selected case and plot tab,
telemetry CSV path/color/display/column selection/time shift, and window layout.
Settings are saved after edits and when the window closes, then restored on the
next launch from `Results/gui/gui_state.json`. Telemetry is reloaded from its CSV;
if that file has moved, select it again with **Load telemetry CSV…**. Closing during
optimization cancels the active solve and retains the previous completed result;
reopening never starts an optimization automatically.

**Reset GUI** restores all three cases and display settings to their defaults and
clears the current plots and telemetry selection. The reset is saved immediately.
Previously saved optimization run folders remain on disk. **Reset dispersion to
defaults** still resets only the selected case's dispersion factors.

Click **Fit dispersion to telemetry…** to search for factors that improve the
altitude and ECEF-speed comparison. The dialog selects which factors to vary,
absolute lower/upper bounds, solver-evaluation budget (80 by default), altitude
and speed error scales, and the destination case. Thrust and Isp multipliers for
both ascent stages are selected initially. The default destination is an empty
comparison slot, so the source case remains available alongside the fit.

Every successful candidate is synchronized independently **before comparison**:
the first simulated stage-one speed is matched to the first rising interpolated
telemetry crossing. Stage-two speed is used if first-stage speed is unavailable.
Shifts are always recalculated from original telemetry times. A candidate that
cannot be synchronized is rejected. The existing manual telemetry shift does
not override this automatic per-candidate synchronization.

The search holds payload fixed. A positive payload entry is used directly;
maximize mode uses the source case's computed payload, or runs a preliminary
nominal optimization to determine it. It then uses bounded Powell search,
robust normalized altitude/speed errors balanced across channels and phases,
and a configurable penalty for changes from the starting factors. Comparison
samples are fixed from baseline overlap (with a small boundary margin). Missing
coverage is penalized; below 90% coverage in any channel the candidate is rejected.
Derived acceleration, pressure, and heat flux are not included in the score.

The GUI stays responsive, displays evaluation progress and best score, and saves
improving candidates throughout the search. **Cancel** stops the worker and loads
its best completed comparison, if one exists. Without a successful comparison,
the destination's previous result is retained. **Fit results** displays fitted
values, bounds, baseline/best errors, synchronization, and evaluation history.
The best candidate appears in the destination case's color on the usual plots,
and telemetry is aligned to that candidate. Fitted values, reports, and search
settings are saved with the GUI session. Run artifacts include `fit_inputs.json`
(with a telemetry snapshot), `fit_report.json`, `fit_history.json`,
`fit_best.pickle`, and the best `solution.pickle`/`inputs.json` under `Results/gui/`.
Closing the GUI retains checkpoint files; it does not resume a search on startup.

The standalone function is in [DispersionFit.py](DispersionFit.py):

```python
from DispersionFit import fit_dispersion_to_telemetry

fit = fit_dispersion_to_telemetry(
    base_request, telemetry,
    parameters=["FirstStageThrust", "SecondStageThrust"],
    bounds={"FirstStageThrust": (0.9, 1.1), "SecondStageThrust": (0.9, 1.1)},
    max_evaluations=40,
)
best_values = fit["best"]["request"]["dispersion"]
best_simulation = fit["best"]["solution"]
```

The function also accepts progress/checkpoint/cancellation callbacks, an initial
solution for warm starts, explicit phase windows, and solver options. Fitting
finds a best candidate within its evaluation budget; correlated factors and
changes in optimized steering can prevent unique physical identification.

Each computed case displays its payload mass and **S2 propellant left (est.)**.
This estimates propellant remaining after the modeled final orbit-adjustment
burn and payload separation: final `x3` mass minus payload, dispersed second-stage
dry mass, and `propellant_mass_for_final_dv`. Separation itself consumes no fuel
in the model. The value uses the computed case's vehicle, stage partition, and
dry-mass offsets, so editing inputs does not change it before recomputation.
Older results without the final-burn estimate display `unavailable`.

Editing a case keeps its last computed curves visible, with an explicit edited
status. A successful recomputation replaces only that case's curves. Failed or
cancelled recomputations retain its previous result and leave other cases
untouched. Plot summaries and target lines always describe the computed inputs.
First-stage steering uses dashed lines; orbital apogee/perigee use solid/dashed
lines, with dotted/dash-dot target lines in each case's color.

Use **Load telemetry CSV…** above the plots to select a flight telemetry file.
Telemetry uses a separate color and dashed lines with sample markers; its
**Display telemetry**, **Color**, and **Clear** controls are independent of the
three simulation cases. Loading a replacement file preserves the simulations.
Recognized measurements appear on their matching axes, including the detail
tabs; missing samples remain gaps. Telemetry can also be viewed before solving.

Telemetry acceleration is calculated from ECEF speed using numerical differences:
`a[i] = (v[i] - v[i-1]) / (t[i] - t[i-1])`, in m/s² at the later sample time.
It appears as **Telemetry · dv/dt (ECEF)** on **S2 propulsion** and **Booster**,
alongside any acceleration supplied in the CSV. First-stage telemetry is not
numerically differentiated; **S1 details** keeps its supplied CSV acceleration. Time steps may
be unequal; the first sample, missing/invalid data, and zero-duration intervals
remain gaps. Values retain their sign during deceleration; no smoothing is applied.
On **S2 propulsion**, simulation curves also use the signed rate of change of
speed from the full velocity history, including gravity. With velocity telemetry
shown, simulation velocities are converted to ECEF before differentiation so both
curves share the same reference frame; otherwise simulation uses ECI. Each burn
is differentiated separately across fairing separation. This is tangential
acceleration (`d|v|/dt`), not the norm of the acceleration vector. First-stage and
booster simulation **Specific acceleration** excludes gravity. Both endpoints
must fall inside the plotted flight-phase window.

When altitude and speed samples are both available, stage-one and booster
telemetry dynamic pressure is calculated as `q = 0.5 * rho(h) * V²` and shown
in kPa. The **Booster** tab also includes an empirical heat-flux plot for both
simulation and telemetry: `heat_flux = k * sqrt(rho(h)) * V³`, in kW/m², using
the vehicle model's heating coefficient (currently `k = 2e-4`). These calculations
use the project's log-interpolated atmosphere and Earth-fixed speed as the
no-wind atmosphere-relative speed. Telemetry estimates use nominal density;
each simulated case includes its own `AtmosphereDensity` dispersion.

Calculated curves are labelled **Telemetry · derived**. Missing altitude/speed
samples and invalid speeds remain gaps. Supplied `dynamic_pressure_kpa`/`_pa`
and `heat_flux_kw_m2`/`heat_flux_w_m2` readings remain separate **Telemetry · CSV**
curves, so they can also be compared with the calculated estimates. Stage
selection, time shifts, display toggles, and case colors apply to these plots.

The loader supports all three CSV formats in `falcon9_telemetry/`, including
`time_str_primary`, `speed_kmh`, `altitude_km`, `acceleration_g`, the raw extraction
columns, and the separate `velocity2_kmh` / `altitude2_km` channels. Other files
can use `time_s` or `mission_time_s` and unit-labelled measurements. Examples
include `speed_mps`, `altitude_m`, `mass_kg`, `thrust_n`, `thrust_factor`,
`angle_of_attack_deg`, `dynamic_pressure_kpa`, `isp_s`, and
`specific_acceleration_mps2`. The full recognized column list is in
`TelemetryCSV.py` (`CHANNELS`). Stage-specific columns can use `stage1_`,
`stage2_`, or `booster_` prefixes (also `s1_` / `s2_`).

**Primary columns** assigns unprefixed readings to stage 1/booster, stage 2, or
booster. Explicit stage columns take precedence. When simulations are displayed,
stage-one and booster readings are limited to the union of their flight time
ranges. Mission timestamps support signed `T±HH:MM:SS` and fractional seconds.
Missing mission times use video times aligned to available mission timestamps;
video-only files start at zero and are labelled as video-relative. Enter a
**Shift [s]** and click **Apply** to align telemetry; positive shifts move it later.

Select a computed case and click **Sync to Case N** to align telemetry
by its initial first-stage speed. The GUI finds the first rising telemetry
speed crossing, linearly interpolates its timestamp, and sets
`shift = case_initial_time - interpolated_telemetry_time`. The same shift
applies to all telemetry channels and flight phases. Repeated clicks recalculate
from original CSV times, so shifts do not accumulate. The selected case's
computed result supplies the reference even if its inputs have since been edited.
If no matching ascent interval exists, the GUI reports the issue and preserves
the previous shift. Missing speed samples are skipped for this alignment only;
the plotted data retain their gaps. Select **Stage 1 / booster** for unprefixed
first-stage speeds; explicitly prefixed first-stage columns are also supported.

Velocity telemetry is treated as **ECEF**. When matching velocity telemetry
is displayed, simulated stage-one and booster velocities are rotated from the
launch-local frame to ECEF using each case's launch azimuth. Their speed
magnitudes are unchanged by this rotation. Simulated stage-two velocities are
converted from ECI using `v_ECEF = Rz(-omega*t) * (v_ECI - omega × r_ECI)`.
The Earth rotation term is evaluated at the vehicle's position, not the launch
site. Telemetry speeds are never given an assumed inertial correction.

Comparison speed plots are labelled **ECEF speed**, and stage-one velocity
component plots use ECEF X/Z. CSV vector columns may be named `ecef_vx_mps`,
`ecef_vy_mps`, `ecef_vz_mps` (or `vx_mps`, `vy_mps`, `vz_mps`), with the usual
stage prefixes. Complete vectors provide speed by their norm; missing vector
samples can fall back to the CSV's scalar speed. The sync button likewise
compares ECEF speed. Hiding or clearing telemetry restores the simulation's
original local/ECI plots. Orbital diagnostics and explicitly labelled ECI thrust
quantities retain their original definitions. Pressure and heat-flux estimates
continue to use the original ECEF/atmosphere-relative speed.

Altitudes are above the model's Earth reference surface; orbital apogee/perigee
CSV columns are altitudes, and acceleration in g converts using 9.80665 m/s².
Angles and thrust components must use the frame labelled on the plot.

Results appear in **First stage** and **Second stage** tabs. Selecting ASDS
(drone ship) or RTLS (return to launch site) adds a **Booster** tab when its result is displayed.
Additional **S1 details**, **S2 propulsion**, and **S2 orbit** tabs show:

- Stage 1: specific impulse, specific acceleration, angle of attack, local
  horizontal/vertical velocities, and flight angle relative to the local X axis.
- Stage 2 propulsion: specific impulse, speed acceleration including gravity, thrust-to-ECI-
  velocity angle, and the three ECI thrust components.
- Stage 2 orbit: geocentric radius, radial velocity, transverse speed, flight
  angle relative to the local horizon, eccentricity, and specific orbital energy.

All detail tabs overlay the displayed cases using their assigned colors. Stage 2
curves cover both sides of fairing separation. Specific acceleration excludes
gravity; orbital diagnostics use the model's Earth gravitational parameter.
Undefined angles at zero speed or zero thrust appear as gaps in the curves.

The vehicle selector also supports Starship. Each tab has Matplotlib controls
for zooming, panning, and saving plots. Settings are locked during a run; the
solver runs in a separate process, with live output and a **Cancel** button.
Only converged solutions are plotted; failed runs display their solver status.
Each run saves its exact inputs, solver log, and result pickle in a unique
folder under `Results/gui/`. The window requires a desktop display and Python
Tk bindings; no additional pip packages are needed.

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
