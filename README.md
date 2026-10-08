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
Each dispersion name shows its nominal vehicle quantity in the main panel and
fit dialog: Isp in seconds (`s`), thrust in metric ton-force (`tf`), and mass in
metric tons (`t`). First-stage propulsion shows both sea-level (SL) and vacuum
values. These annotations follow the selected vehicle and launch-site altitude;
the editable factors and mass offsets retain their existing units.

Click **New vehicle…** beside the vehicle selector to create a vehicle from the
currently selected vehicle's nominal parameters. The editor includes geometry,
dry/propellant/fairing masses, sea-level and vacuum propulsion, aerodynamic
coefficients and limits, booster recovery loads, and flight limits. Enter masses
in metric tons, thrust in metric ton-force, and Isp in seconds. Give the vehicle
a unique name and click **Create vehicle** to add it to the selector and select
it for the active case. Existing cases retain their own vehicle selections.
Custom vehicles can also be used as templates for another new vehicle.

The **3D trajectory** tab shows first- and second-stage ascent in ECI coordinates,
Earth, the equatorial plane, and parking/final orbits, with XY, YZ, and XZ
projections as in `Main.py`. Drag the 3D view to rotate it and use the plot
toolbar to save the figure. Case colors and **Display** selections also apply
to this tab. Orbit shapes are sampled as two-body ellipses; the final orbit
uses the saved speed change at parking apogee. Unbound states show the ascent
with an explanation instead of an orbit. Scalar altitude/speed telemetry has
no 3D position information and is shown on its existing flight-phase tabs.
The first-stage path uses the saved launch-site/azimuth transform, matching the
solver's stage-separation frame. Each displayed case has its own results box
with matching text and border colors: payload, launch azimuth (east from north),
parking-to-final-orbit Δv at apogee, orbit apogee/perigee and inclination, MECO
and SECO mission times, maximum ascent dynamic pressure, and orbit-burn propellant.

Vehicle definitions are saved with the GUI state and included in optimization
and telemetry-fit requests and results. The editor's **Context** tab shows the
fixed Earth environment and explains which quantities are calculated. Launch
site, target orbit, payload, and dispersion remain case settings; the existing
flight equations and recovery engine fractions also apply to custom vehicles.

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

Use **Save state…** to export the same complete snapshot to a chosen JSON file,
and **Load state…** to restore it later. Loading a state file also updates the
automatic startup snapshot. A corrupt or incompatible file leaves the current
GUI state unchanged.

**Reset GUI** restores all three cases and display settings to their defaults and
clears the current plots and telemetry selection. The reset is saved immediately.
Previously saved optimization run folders remain on disk. **Reset dispersion to
defaults** still resets only the selected case's dispersion factors.

Click **Fit dispersion to telemetry…** to search for factors that improve the
altitude, ECEF speed, acceleration, and dynamic-pressure comparison. The dialog
selects which factors to vary, absolute lower/upper bounds, solver-evaluation
budget (100 by default in the GUI), four error scales, and the destination case.
The initial factor selection is defined by `DEFAULT_PARAMETERS` in `DispersionFit.py`.
The default destination is an empty
comparison slot, so the source case remains available alongside the fit.
Each time the dialog opens, lower and upper bounds reset to the vehicle's nominal
defaults. The initial dispersion guess uses the selected case's current values;
its computed trajectory, when available, supplies the solver's initial guess.
Bounds can be edited for that search and must contain the initial values.

Every successful candidate is synchronized independently **before comparison**:
the first simulated stage-one speed is matched to the first rising interpolated
telemetry crossing. Stage-two speed is used if first-stage speed is unavailable.
Shifts are always recalculated from original telemetry times. A candidate that
cannot be synchronized is rejected. The existing manual telemetry shift does
not override this automatic per-candidate synchronization.

The search holds payload fixed. A positive payload entry is used directly;
maximize mode uses the source case's computed payload, or runs a preliminary
nominal optimization to determine it. It then uses bounded Powell search,
robust normalized errors balanced across available channels and phases,
and a configurable penalty for changes from the starting factors. Comparison
samples are fixed from baseline overlap (with a small boundary margin). First-
and second-stage comparisons use only times with positive telemetry acceleration
(`dv/dt > 0`, backward differences of ECEF speed). The same time mask applies to
all ascent channels; zero/negative acceleration and missing speed intervals
are excluded. This omits cutoff and coast samples before
second-stage ignition. Each stage needs speed telemetry to select its samples.
For **ASDS and RTLS**, available booster altitude and speed telemetry after
separation also contribute to the score, covering the return trajectory through
landing (including RTLS boostback). Booster samples include coasting and
deceleration; either altitude or speed can be used independently. **EXP** excludes
booster telemetry. All phases use the same per-candidate synchronization, which
still requires first- or second-stage speed telemetry. Booster errors receive the
same phase weighting as each ascent stage and appear in the fit report.
Missing coverage is penalized; below 90% coverage in any channel the candidate is rejected.
Rows with no usable telemetry measurement are skipped, including internal gaps;
they add neither a measurement residual nor a missing-coverage penalty. Missing
coverage means the simulation cannot predict an existing telemetry measurement.
Each fitted phase also receives a duration penalty in either direction and an
extra penalty for simulation samples later than that phase's last telemetry
reading. Timing uses the full first-to-last observed phase extent, including
coasts and cutoff samples, rather than the trimmed positive-acceleration fit
window. Empty leading/trailing rows do not extend a phase. These telemetry
bounds are fixed before searching. When unlabelled primary columns are shared
by first-stage ascent and booster return, the baseline separation time divides
them; explicitly labelled phase columns retain their complete observed extent.
**Phase timing error scale [s]** defaults to **5**; smaller values strengthen
both timing penalties. The fit report shows telemetry and simulation durations,
their difference, late sample counts, and end overruns for each fitted phase.
Acceleration and dynamic pressure contribute during **first- and second-stage
ascent only**, with the same robust loss as altitude and speed. These channels
are ignored for booster return and other phases, whether measured or calculated.
Finite supplied ascent telemetry takes precedence, with missing
values calculated where possible. Supplied acceleration follows the CSV's
specific-acceleration convention and is compared to the simulation's
non-gravitational force per unit mass (including aerodynamic forces).
Calculated acceleration uses signed backward differences of ECEF speed; the
simulation comparison uses the same telemetry time intervals, including gravity.
This keeps the two acceleration definitions separate
without counting both at one timestamp. Dynamic pressure is calculated as
`½ρv²` using Earth-fixed speed and the project's atmosphere model. Telemetry
calculations use nominal density; simulation calculations include that case's
atmospheric-density dispersion, including stage two.

The fit dialog exposes **Acceleration error scale [m/s²]** (default **1**) and
**Dynamic pressure error scale [kPa]** (default **5**), alongside altitude
(1000 m) and speed (25 m/s). Smaller scales give stronger mismatch penalties;
all scales must be finite and positive. These settings persist with the case
and appear in the fit report, together with measured/calculated sample counts.
Heat flux is excluded from the fit score.

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

Telemetry extraction and the revised files in `falcon9_telemetry/` use three
independent phase groups:

| Phase | CSV columns |
| --- | --- |
| Stage 1, before separation | `velocity1_kmh`, `altitude1_km`, `acceleration1_g` |
| Stage 2, upper stage | `velocity2_kmh`, `altitude2_km` |
| Booster return, ASDS/RTLS | `velocity3_kmh`, `altitude3_km` |

The first two columns are `t_video_sec` and `time_str_primary`. In
`ExtractTelemetry.py`, `START_SEP_SEC` (or CLI `--separation`) is the video time
at which stage 1 ends. Samples before that time populate group 1; samples at or
after it populate groups 2 and 3 concurrently. Inactive groups stay blank;
EXP leaves group 3 blank throughout. The extraction preview plots show separate
stage-one, upper-stage, and booster velocity/altitude curves.

`TELEMETRY_ROI_PHASE3` defines the upper-stage-only layout after booster landing.
Set `START_PHASE3_SEC` or pass `--landing SECONDS` (alias `--phase3-start`) to
switch layouts at that original broadcast timestamp. It defaults to `None`
until the landing/layout-change time is configured. At and after this time,
only `velocity2`, `altitude2`, and `time` are read; upper-stage measurements
continue in `velocity2_kmh` / `altitude2_km`, and booster columns stay blank.
Absent booster gauges no longer cause OCR correction prompts. Phase 3 defaults
to the left-side speed/altitude crops; tune its independent ROI coordinates
for the broadcast. `--preview --start SECONDS --landing SECONDS` uses the same
layout selection as extraction. Landing time must not precede separation.

The GUI automatically selects **Phase columns** for these files. The loader,
plots, synchronization, and fit use the numbered columns directly, so stage-one
readings never fill gaps in the upper-stage or booster histories. Fitting uses
supplied stage-one acceleration; upper-stage acceleration and ascent dynamic
pressure are calculated where no measurement is supplied. Acceleration and
pressure penalties apply only to ascent. Booster altitude/speed contribute for
ASDS/RTLS; EXP ignores booster readings.

Older unnumbered and raw extraction columns remain supported, including
`speed_kmh`, `altitude_km`, and `acceleration_g`. Other files
can use `time_s` or `mission_time_s` and unit-labelled measurements. Examples
include `speed_mps`, `altitude_m`, `mass_kg`, `thrust_n`, `thrust_factor`,
`angle_of_attack_deg`, `dynamic_pressure_kpa`, `isp_s`, and
`specific_acceleration_mps2`. The full recognized column list is in
`TelemetryCSV.py` (`CHANNELS` and `NUMBERED_CHANNELS`). Stage-specific columns can use `stage1_`,
`stage2_`, or `booster_` prefixes (also `s1_` / `s2_`).

**Column routing** selects phase columns directly or assigns legacy unprefixed
readings to stage 1/booster, stage 2, or booster. Explicit numbered/prefixed stage
columns take precedence. With **Phase columns** routing, telemetry displays its
full observed history, including booster landing readings later than the simulated landing.
Legacy primary readings shared between stage one and booster are limited to the
union of the displayed simulations' respective flight time ranges for routing.
Mission timestamps support signed `T±HH:MM:SS` and fractional seconds.
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

The vehicle selector also supports Starship and Starship V3 (configuration 3).
Each tab has Matplotlib controls
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

For `ExtractTelemetry.py`, `START_TIME_SEC` and `END_TIME_SEC` also select the
download interval, so a new `VIDEO_FILENAME` contains only that part of the
broadcast. Override them with `--start` / `--end` or `--t1` / `--t2`:

```bash
PYTHONPATH= .venv/bin/python ExtractTelemetry.py \
  --t1 604 --t2 1130 --separation 755
```

Without `--video`, this downloads the configured `VIDEO_URL` directly into
`VIDEO_FILENAME` using [yt-dlp's time-range download option](https://github.com/yt-dlp/yt-dlp#usage-and-options)
and ffmpeg. The adjacent `.mp4.clip.json` file records the clip's original start
and end times. Keep it with the clip: CSV `t_video_sec` values and `--separation`
continue to use the original broadcast clock, even though the clip starts at
local time zero. Preview also seeks within the clip correctly. A cached clip
can be reused for any interval it contains; use another `VIDEO_FILENAME` when
requesting a different broadcast or an interval outside that clip. Existing
videos without clip metadata are treated as full videos and reused without
being overwritten. `--video path/to/video.mp4` uses an existing local video.
The telemetry CSV and optional `_raw.csv` use that video's basename under
`--output-dir`.
