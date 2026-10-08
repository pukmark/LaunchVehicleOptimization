#!/usr/bin/env python3
"""Desktop frontend for one launch-vehicle optimization at a time.

Run with PYTHONPATH= .venv/bin/python OptimizationGUI.py.
The private --worker entry point runs IPOPT outside the Tk event loop.
"""
import argparse
from dataclasses import asdict, dataclass, field, fields
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from DispersionFit import DEFAULT_PARAMETERS, ERROR_SCALES, CHANNEL_UNITS, default_fit_bounds, fit_report
from LaunchSites import LAUNCH_SITES, DEFAULT_LAUNCH_SITE, launch_site_inputs, parse_launch_site
from GUIState import read_gui_state, write_gui_state
from VehicleDefinitions import (BUILTIN_VEHICLES, VEHICLE_PARAMETER_GROUPS, vehicle_catalog,
                                vehicle_editor_values, vehicle_from_editor, vehicle_name)
from LV_Type_scaled import DispesrionFactorsType, VLType
from TelemetryFrames import local_to_ecef_velocity, eci_to_ecef_velocity
from Trajectory3D import draw_trajectory_3d
from TelemetryCSV import (load_telemetry_csv, overlay_telemetry, phase_group,
                          aerodynamic_loads, make_plot_axes, interpolate_speed_time,
                          numerical_acceleration, PRIMARY_GROUPS)


PROJECT_DIR = Path(__file__).resolve().parent
RECOVERY = {"EXP — Expendable": "EXP", "ASDS — Drone ship": "ASDS",
            "RTLS — Return to launch site": "RTLS"}
VEHICLES = BUILTIN_VEHICLES
RESULT_TABS = ("First stage", "S1 details", "Second stage", "S2 propulsion", "S2 orbit", "3D trajectory", "Booster")
MULTIPLIERS = {"FirstStageIsp", "SecondStageIsp", "FirstStageThrust",
               "SecondStageThrust", "FirstStageCx0", "BoosterStageCx0", "AtmosphereDensity"}


def default_case_inputs():
    return {"vehicle": "Falcon 9", "recovery": next(iter(RECOVERY)), "payload_mass": "-1", "launch_site": launch_site_inputs(),
            "dispersion": {k: str(v) for k, v in asdict(DispesrionFactorsType()).items()},
            "orbit": {"apogee": "200.0", "perigee": "200.0", "inclination": "28.6"}}


@dataclass
class ComparisonCase:
    name: str
    color: str
    draft: dict = field(default_factory=default_case_inputs)
    computed_inputs: dict = None
    request: dict = None
    result: dict = None
    run_dir: Path = None
    state: str = "Not run"
    fit_report: dict = None
    fit_settings: dict = None

    @property
    def edited(self):
        return self.computed_inputs is not None and self.draft != self.computed_inputs


def field_unit(name):
    if name in MULTIPLIERS:
        return "×"
    if name.endswith("Mass"):
        return "Δ kg"
    return "Δ m" if name == "LaunchAltDelta" else "Δ fraction"


def nominal_dispersion_values(configuration=1, launch_altitude=0.):
    """Display the vehicle quantities before applying any dispersion."""
    model = VLType()
    model.ApplyVehicleConfiguration(configuration)
    def number(value):
        return f"{value:,.3f}".rstrip('0').rstrip('.')
    tonne_force = 1000. * 9.80665
    try:
        altitude = float(launch_altitude)
    except (TypeError, ValueError):
        altitude = np.nan
    return {
        'FirstStageIsp': f"SL {number(model.FirstStage_SL_Isp)} / vac {number(model.FirstStage_Vac_Isp)} s",
        'SecondStageIsp': f"{number(model.SecondStage_Vac_Isp)} s",
        'FirstStageThrust': (f"SL {number(model.FirstStage_SL_Thrust / tonne_force)} / "
                             f"vac {number(model.FirstStage_Vac_Thrust / tonne_force)} tf"),
        'SecondStageThrust': f"{number(model.SecondStage_Thrust / tonne_force)} tf",
        'FirstStageCx0': number(model.FirstStage_Cd0),
        'BoosterStageCx0': number(model.Booster_Cd0),
        'FirstStage_EmptyMass': f"{number(model.EmptyFirstStageMass / 1000.)} t",
        'SecondStage_EmptyMass': f"{number(model.EmptySecondStageMass / 1000.)} t",
        'FirstStage_PropMass': f"{number(model.FirstStagePropellentMass / 1000.)} t",
        'SecondStage_PropMass': f"{number(model.SecondStagePropellentMass / 1000.)} t",
        'AtmosphereDensity': '1 × density profile',
        'LaunchAltDelta': f"{number(altitude)} m" if np.isfinite(altitude) else '— m',
        'StagePartitionDelta': f"{model.FirstStagePropellentMass / model.TotalPropellentMass:.4f} S1 fraction",
    }


def parse_dispersion(values, configuration=1):
    """Validate the entire form before launching a potentially long solve."""
    parsed = {}
    for item in fields(DispesrionFactorsType):
        try:
            value = float(values[item.name])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f"{item.name}: enter a number.") from None
        if not np.isfinite(value):
            raise ValueError(f"{item.name}: enter a finite number.")
        if item.name in MULTIPLIERS and value <= 0:
            raise ValueError(f"{item.name}: the multiplier must be greater than zero.")
        parsed[item.name] = value
    dispersion = DispesrionFactorsType(**parsed)
    model = VLType()
    model.ApplyVehicleConfiguration(configuration)
    partition = model.FirstStagePropellentMass / model.TotalPropellentMass
    if not 0 < partition + dispersion.StagePartitionDelta < 1:
        raise ValueError("StagePartitionDelta must leave propellant in both stages.")
    model.ApplyScenarioDispersionToLV(dispersion)
    for name in ("EmptyFirstStageMass", "EmptySecondStageMass",
                 "FirstStagePropellentMass", "SecondStagePropellentMass"):
        if not np.isfinite(getattr(model, name)) or getattr(model, name) <= 0:
            raise ValueError(f"Mass and partition offsets must keep {name} positive and finite.")
    if model.R0 + model.LaunchAltitude <= 0:
        raise ValueError("LaunchAltDelta places the launch site at or below Earth's center.")
    return dispersion


def parse_target_orbit(values=None):
    """Convert user-facing altitudes (km) and inclination (deg) to solver units."""
    if values is None:
        values = {"apogee": 200.0, "perigee": 200.0, "inclination": 28.6}
    parsed = {}
    for name in ("apogee", "perigee", "inclination"):
        try:
            value = float(values[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Target {name}: enter a number.") from None
        if not np.isfinite(value):
            raise ValueError(f"Target {name}: enter a finite number.")
        parsed[name] = value
    if parsed["perigee"] <= 0 or parsed["apogee"] <= 0:
        raise ValueError("Target apogee and perigee altitudes must be above zero km.")
    if parsed["apogee"] < parsed["perigee"]:
        raise ValueError("Target apogee must be greater than or equal to perigee.")
    if not 0 <= parsed["inclination"] <= 180:
        raise ValueError("Target inclination must be between 0 and 180 degrees.")
    apogee = 6378137.0 + 1000 * parsed["apogee"]
    perigee = 6378137.0 + 1000 * parsed["perigee"]
    if not np.isfinite(apogee) or not np.isfinite(perigee):
        raise ValueError("Target orbit altitudes are too large.")
    nominal = parsed == {"apogee": 200.0, "perigee": 200.0, "inclination": 28.6}
    return {"Name": "LEO" if nominal else "Custom", "apogee": apogee,
            "perigee": perigee, "i": float(np.deg2rad(parsed["inclination"]))}


def parse_payload_mass(value):
    try:
        mass = float(value)
    except (TypeError, ValueError):
        raise ValueError("Payload mass: enter a number in kg (0 or negative to maximize).") from None
    if not np.isfinite(mass):
        raise ValueError("Payload mass: enter a finite number.")
    return mass


def nominal_request(values, configuration=1, recovery="EXP", orbit_values=None, payload_mass=-1, launch_site=None):
    if recovery not in RECOVERY.values():
        raise ValueError("Unknown recovery strategy.")
    site = parse_launch_site(launch_site)
    dispersion = parse_dispersion(values, configuration)
    if 6378137.0 + site["altitude"] + dispersion.LaunchAltDelta <= 0:
        raise ValueError("Launch altitude plus LaunchAltDelta must be above Earth's center.")
    return {
        "launch_site": site,
        "dispersion": asdict(dispersion),
        "configuration": deepcopy(configuration),
        "recovery": recovery,
        "target_orbit": parse_target_orbit(orbit_values),
        "payload_mass_predefined": parse_payload_mass(payload_mass),
    }


def run_optimization(request, output_dir, ipopt_options=None):
    """Small solver boundary, also usable without a desktop display."""
    from LV_Optimization_Type import LV_Optimization

    dispersion = parse_dispersion(request["dispersion"], request["configuration"])
    model = LV_Optimization(dispersion, request["configuration"],
                            launch_site=parse_launch_site(request.get("launch_site")))
    result = model.SolveOptimiztion(
        RecoveryStrategy=request["recovery"],
        payload_mass_predefined=parse_payload_mass(request.get("payload_mass_predefined", -1)),
        Target_Orbit=request["target_orbit"], Plot_interm=False,
        print_ipopt=True, output_dir=output_dir, ipopt_options=ipopt_options,
    )
    return result


def worker(directory):
    """Persist the result even when IPOPT reports a failed solve."""
    directory = Path(directory)
    request = json.loads((directory / "inputs.json").read_text())
    result = run_optimization(request, output_dir=None)
    name = "solution.pickle" if result.get("success") else "solution_Failed.pickle"
    with (directory / name).open("wb") as stream:
        pickle.dump(result, stream)


def second_stage_propellant_remaining(solution, request=None):
    """Estimated propellant after the modeled final burn and payload separation.

    x3's final mass includes dry stage, payload, and remaining propellant before
    the reserved final orbit-adjustment burn. Separation itself consumes no fuel
    in this model. Use computed metadata, never the currently edited form.
    """
    if "propellant_mass_for_final_dv" not in solution:
        return None
    request = request or {}
    model = VLType()
    configuration = solution.get("configuration", request.get("configuration", 1))
    model.ApplyVehicleConfiguration(configuration)
    dispersion = solution.get("dispersion", request.get("dispersion", {}))
    model.ApplyScenarioDispersionToLV(DispesrionFactorsType(**dispersion))
    remaining = (float(np.asarray(solution["x3"])[6, -1])
                 - float(solution["payload_mass"]) - model.SecondStage_EmptyMass
                 - float(solution["propellant_mass_for_final_dv"]))
    if not np.isfinite(remaining):
        return None
    # Avoid displaying negative zero from solver tolerances at 0.1 kg precision.
    return 0.0 if abs(remaining) < 0.05 else remaining


def remaining_propellant_text(case):
    remaining = second_stage_propellant_remaining(case.result, case.request)
    return "unavailable" if remaining is None else f"{remaining:,.1f} kg"


def orbital_diagnostics(state):
    """Derive osculating quantities from SI position/velocity in the ECI frame."""
    position, velocity = np.asarray(state)[:3], np.asarray(state)[3:6]
    radius = np.linalg.norm(position, axis=0)
    radial = np.divide(np.sum(position * velocity, axis=0), radius,
                       out=np.full_like(radius, np.nan, dtype=float), where=radius > 0)
    angular_momentum = np.cross(position.T, velocity.T).T
    transverse = np.divide(np.linalg.norm(angular_momentum, axis=0), radius,
                           out=np.full_like(radius, np.nan, dtype=float), where=radius > 0)
    unit_radius = np.divide(position, radius, out=np.full_like(position, np.nan, dtype=float), where=radius > 0)
    eccentricity = np.linalg.norm(np.cross(velocity.T, angular_momentum.T).T / VLType.mu - unit_radius, axis=0)
    potential = np.divide(VLType.mu, radius, out=np.full_like(radius, np.nan, dtype=float), where=radius > 0)
    return {"radius": radius, "radial": radial, "transverse": transverse,
            "flight_path": np.where(np.linalg.norm(velocity, axis=0) > 0,
                                   np.rad2deg(np.arctan2(radial, transverse)), np.nan),
            "eccentricity": eccentricity,
            "energy": 0.5 * np.sum(velocity**2, axis=0) - potential}


def thrust_velocity_angle(state, control):
    """Thrust-to-inertial-velocity angle in degrees; undefined at zero thrust/speed."""
    velocity = np.asarray(state)[3:6, :-1]
    thrust = np.asarray(control)
    denominator = np.linalg.norm(velocity, axis=0) * np.linalg.norm(thrust, axis=0)
    cosine = np.divide(np.sum(velocity * thrust, axis=0), denominator,
                       out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def local_velocity_in_ecef(solution, velocity):
    site = parse_launch_site(solution.get("launch_site"))
    return local_to_ecef_velocity(velocity, float(solution["LaunchAz"]),
                                  site["latitude"], site["longitude"])


def stage_two_speed_acceleration(solution, number, speed_frame="Inertial"):
    """Differentiate simulated speed, whose dynamics already include gravity.

    Use the same signed dv/dt as scalar telemetry, not the magnitude of the
    acceleration vector. Differentiate each burn separately to avoid crossing
    the fairing-separation boundary. Convert velocity before differentiation.
    """
    time = np.asarray(solution[f"t{number}_vec"]).ravel()
    state = np.asarray(solution[f"x{number}"])
    velocity = state[3:6]
    if speed_frame == "ECEF":
        velocity = eci_to_ecef_velocity(state[:3], velocity, time)
    return numerical_acceleration(time, np.linalg.norm(velocity, axis=0))


def draw_phase(figure, phase, solution, target, *, color="tab:blue", case_label="Case", append=False, speed_frame="Inertial"):
    """Plot SI solver output; return states are samples × states, unlike ascent."""
    if not append:
        figure.clear()
        axes = make_plot_axes(figure, phase)
    else:
        axes = figure.axes[:7 if phase == "Booster" else 6]

    def line(index, x, y, title, ylabel, label=None, xlabel="Time [s]", **kwargs):
        ax = axes[index]
        legend_label = f"{case_label} · {label}" if label else case_label
        ax.plot(np.asarray(x).ravel(), np.asarray(y).ravel(), linewidth=1.6,
                label=legend_label, color=color, **kwargs)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=0.25)

    s = solution
    if phase == "First stage":
        t, x = np.asarray(s["t1_vec"]).ravel(), np.asarray(s["x1"])
        line(0, x[0] / 1000, x[1] / 1000, "Local trajectory", "Local Z [km]", xlabel="Downrange [km]")
        line(1, t, s["alt1_sol"] / 1000, "Altitude", "Altitude [km]")
        velocity = x[2:4]
        if speed_frame == "ECEF":
            velocity = local_velocity_in_ecef(s, np.vstack((x[2], np.zeros_like(x[2]), x[3])))
        line(2, t, np.linalg.norm(velocity, axis=0), "ECEF speed" if speed_frame == "ECEF" else "Local velocity", "Speed [m/s]")
        line(3, t, x[4] / 1000, "Vehicle mass", "Mass [t]")
        line(4, t[:-1], s["u1"][0], "Engine & steering", "Thrust factor [−]", label="Thrust factor")
        steering = figure.axes[6] if append else axes[4].twinx()
        steering.plot(t[:-1], np.rad2deg(s["u1"][1]), "--", color=color,
                      label=f"{case_label} · AoA")
        steering.set_ylabel("Angle of attack [deg] (dashed)")
        line(5, t, s["Qdyn1_sol"] / 1000, "Dynamic pressure", "Pressure [kPa]")
    elif phase == "S1 details":
        t, x = np.asarray(s["t1_vec"]).ravel(), np.asarray(s["x1"])
        line(0, t[:-1], s["Isp1_sol"], "Specific impulse", "Isp [s]")
        line(1, t[:-1], s["acc1_sol"], "Specific acceleration", "Acceleration [m/s²]", label="Specific acceleration")
        line(2, t[:-1], np.rad2deg(s["u1"][1]), "Angle of attack", "Angle [deg]")
        if speed_frame == "ECEF":
            velocity = local_velocity_in_ecef(s, np.vstack((x[2], np.zeros_like(x[2]), x[3])))
            line(3, t, velocity[0], "ECEF velocity X", "Velocity X [m/s]")
            line(4, t, velocity[2], "ECEF velocity Z", "Velocity Z [m/s]")
        else:
            line(3, t, x[2], "Local horizontal velocity", "Velocity X [m/s]")
            line(4, t, x[3], "Local vertical velocity", "Velocity Z [m/s]")
        gamma = np.rad2deg(np.arctan2(x[3], x[2]))
        gamma = np.where(np.linalg.norm(x[2:4], axis=0) > 0, gamma, np.nan)
        line(5, t, gamma, "Flight angle to local X axis", "Angle [deg]")
    elif phase == "S2 propulsion":
        for number in (2, 3):
            t, x, u = np.asarray(s[f"t{number}_vec"]).ravel(), s[f"x{number}"], s[f"u{number}"]
            line(0, t[:-1], s[f"Isp{number}_sol"], "Specific impulse", "Isp [s]")
            frame = "ECEF" if speed_frame == "ECEF" else "ECI"
            acceleration = stage_two_speed_acceleration(s, number, speed_frame)
            line(1, t, acceleration, f"Speed acceleration ({frame}, includes gravity)",
                 "Acceleration [m/s²]", label=f"dv/dt ({frame}, incl. gravity)")
            line(2, t[:-1], thrust_velocity_angle(x, u), "Thrust / ECI velocity angle", "Angle [deg]")
            for index, component in enumerate(("X", "Y", "Z")):
                line(3 + index, t[:-1], u[index] / 1000, f"ECI thrust {component}", "Thrust [kN]")
    elif phase == "S2 orbit":
        for number in (2, 3):
            t = s[f"t{number}_vec"]
            orbit = orbital_diagnostics(s[f"x{number}"])
            line(0, t, orbit["radius"] / 1000, "Geocentric radius", "Radius [km]")
            line(1, t, orbit["radial"], "Radial velocity", "Velocity [m/s]")
            line(2, t, orbit["transverse"], "Transverse speed", "Speed [m/s]")
            line(3, t, orbit["flight_path"], "Flight angle to local horizon", "Angle [deg]")
            line(4, t, orbit["eccentricity"], "Orbital eccentricity", "Eccentricity [−]")
            line(5, t, orbit["energy"] / 1e6, "Specific orbital energy", "Energy [MJ/kg]")
    elif phase == "Second stage":
        for number, label in ((2, "Before fairing separation"), (3, "After fairing separation")):
            t, x = s[f"t{number}_vec"], np.asarray(s[f"x{number}"])
            line(0, t, s[f"alt{number}_sol"] / 1000, "Altitude", "Altitude [km]")
            velocity = x[3:6]
            if speed_frame == "ECEF":
                velocity = eci_to_ecef_velocity(x[:3], velocity, np.asarray(t).ravel())
            line(1, t, np.linalg.norm(velocity, axis=0), "ECEF speed" if speed_frame == "ECEF" else "Inertial velocity", "Speed [m/s]")
            line(2, t, x[6] / 1000, "Vehicle mass", "Mass [t]")
            line(3, np.asarray(t).ravel()[:-1], np.linalg.norm(s[f"u{number}"], axis=0) / 1000,
                 "Engine thrust", "Thrust [kN]")
            line(4, t, np.rad2deg(s[f"i{number}_sol"]), "Orbital inclination", "Inclination [deg]")
            for key, style in (("apogee", "-"), ("perigee", "--")):
                line(5, t, (s[f"{key}{number}_sol"] - 6378137.0) / 1000,
                     "Osculating orbit", "Altitude [km]", key.title(), linestyle=style)
        axes[4].axhline(np.rad2deg(target["i"]), color=color, linestyle=":", alpha=0.65)
        axes[4].legend(fontsize=8)
        for key, style in (("apogee", ":"), ("perigee", "-.")):
            axes[5].axhline((target[key] - 6378137.0) / 1000, color=color,
                           linestyle=style, alpha=0.65)
        axes[5].set_yscale("symlog", linthresh=100)
        axes[5].legend(fontsize=8)
    elif phase == "Booster":
        t, x = np.asarray(s["t4_vec"]).ravel(), np.asarray(s["x4"])
        line(0, x[:, 0] / 1000, x[:, 1] / 1000, "Return trajectory", "Local Z [km]", xlabel="Downrange [km]")
        line(1, t, s["alt4_sol"] / 1000, "Altitude", "Altitude [km]")
        velocity = x[:, 2:4].T
        if speed_frame == "ECEF":
            velocity = local_velocity_in_ecef(s, np.vstack((x[:, 2], np.zeros(len(x)), x[:, 3])))
        line(2, t, np.linalg.norm(velocity, axis=0), "ECEF speed" if speed_frame == "ECEF" else "Local velocity", "Speed [m/s]")
        line(3, t, x[:, 4] / 1000, "Booster mass", "Mass [t]")
        line(4, t, s["acc4_sol"], "Specific acceleration", "Acceleration [m/s²]", label="Specific acceleration")
        line(5, t, s["Qdyn4_sol"] / 1000, "Dynamic pressure", "Pressure [kPa]")
        model = VLType()
        model.ApplyVehicleConfiguration(s.get("configuration", 1))
        loads = aerodynamic_loads(np.asarray(s["alt4_sol"]).ravel(), np.linalg.norm(x[:, 2:4], axis=1),
                                  density_factor=s.get("dispersion", {}).get("AtmosphereDensity", 1.0),
                                  heat_coefficient=model.Booster_k_empirical)
        line(6, t, loads["heat_flux"], "Empirical heat flux", "Heat flux [kW/m²]")
    else:
        raise ValueError(f"Unknown phase: {phase}")
    for ax in axes:
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")


class OptimizationApp:
    def __init__(self, root, output_root=None):
        # Lazy GUI imports allow validation, plotting, and workers without Tk/display.
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.root = root
        self.output_root = Path(output_root) if output_root else PROJECT_DIR / "Results" / "gui"
        self.state_path = self.output_root / "gui_state.json"
        self.save_id = None
        self.layout_id = None
        self.pending_sash = None
        self.restoring_state = True
        self.job_kind = "optimization"
        self.fit_source_case = None
        self.fit_telemetry = None
        self.fit_options = None
        self.process = None
        self.run_dir = None
        self.log_stream = None
        self.cancelled = False
        self.closing = False
        self.started = 0
        self.request = None
        self.result = None
        self.poll_id = None
        self.kill_id = None
        self.cases = [ComparisonCase(f"Case {i + 1}", color) for i, color in
                      enumerate(("#0072B2", "#D55E00", "#009E73"))]
        self.selected_case = 0
        self.vehicles = vehicle_catalog()
        self.running_case = None
        self.loading_case = False
        self.submitted_inputs = None
        self.telemetry = None
        self.telemetry_color = "#CC79A7"
        self.telemetry_shift = 0.0
        self.controls = []
        self.variables = {}
        self.dispersion_labels = {}
        self.figures, self.canvases, self.tabs = {}, {}, {}

        root.title("Launch Vehicle Optimization")
        root.geometry("1440x900")
        root.minsize(1080, 680)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))
        style.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"), padding=10)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        panes = self.panes = ttk.Panedwindow(root, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left, right = ttk.Frame(panes), ttk.Frame(panes, width=380)
        panes.add(left, weight=4)
        panes.add(right, weight=1)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)
        ttk.Label(left, text="Flight simulation", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.summary = tk.StringVar(value="No result yet. Set the factors and run the nominal case.")
        ttk.Label(left, textvariable=self.summary, wraplength=850).grid(row=1, column=0, sticky="w", pady=(0, 8))
        case_panel = ttk.LabelFrame(left, text="Compare cases · select to edit, check to display", padding=6)
        case_panel.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        case_panel.columnconfigure(3, weight=1)
        self.case_selection = tk.IntVar(value=0)
        self.case_visible, self.case_status, self.color_buttons = [], [], []
        for index, case in enumerate(self.cases):
            selector = ttk.Radiobutton(case_panel, text=case.name, variable=self.case_selection,
                                       value=index, command=lambda i=index: self.select_case(i))
            selector.grid(row=index, column=0, sticky="w", padx=4, pady=2)
            self.controls.append((selector, "normal"))
            visible = tk.BooleanVar(value=True)
            ttk.Checkbutton(case_panel, text="Display", variable=visible,
                            command=self.refresh_comparison).grid(row=index, column=1, padx=8)
            color_button = tk.Button(case_panel, text="Color", background=case.color,
                                     foreground="white", activebackground=case.color,
                                     command=lambda i=index: self.choose_color(i), width=6)
            color_button.grid(row=index, column=2, padx=4)
            status = tk.StringVar(value="Not run")
            ttk.Label(case_panel, textvariable=status, wraplength=680).grid(row=index, column=3, sticky="w", padx=8)
            self.case_visible.append(visible)
            self.case_status.append(status)
            self.color_buttons.append(color_button)
        telemetry_panel = ttk.Frame(case_panel)
        telemetry_panel.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        load_csv = ttk.Button(telemetry_panel, text="Load telemetry CSV…", command=self.load_telemetry)
        load_csv.grid(row=0, column=0, sticky="w")
        self.controls.append((load_csv, "normal"))
        self.telemetry_visible = tk.BooleanVar(value=True)
        ttk.Checkbutton(telemetry_panel, text="Display telemetry", variable=self.telemetry_visible,
                        command=self.refresh_comparison).grid(row=0, column=1, padx=6)
        self.telemetry_color_button = tk.Button(telemetry_panel, text="Color", background=self.telemetry_color,
                                               command=self.choose_telemetry_color, width=6)
        self.telemetry_color_button.grid(row=0, column=2, padx=4)
        clear_csv = ttk.Button(telemetry_panel, text="Clear", command=self.clear_telemetry)
        clear_csv.grid(row=0, column=3, padx=4)
        self.controls.append((clear_csv, "normal"))
        self.sync_button = ttk.Button(telemetry_panel, text="Sync to Case 1", command=self.sync_telemetry,
                                      state="disabled")
        self.sync_button.grid(row=0, column=4, padx=(8, 0))
        options = ttk.Frame(case_panel)
        options.grid(row=4, column=0, columnspan=4, sticky="ew", pady=4)
        ttk.Label(options, text="Column routing:").pack(side="left")
        self.telemetry_primary = tk.StringVar(value="Stage 1 / booster")
        primary = ttk.Combobox(options, textvariable=self.telemetry_primary, state="readonly", width=17,
                               values=PRIMARY_GROUPS)
        primary.pack(side="left", padx=4)
        self.controls.append((primary, "readonly"))
        primary.bind("<<ComboboxSelected>>", lambda event: self.refresh_comparison())
        ttk.Label(options, text="Shift [s]:").pack(side="left", padx=(6, 0))
        self.telemetry_shift_input = tk.StringVar(value="0")
        shift = ttk.Entry(options, textvariable=self.telemetry_shift_input, width=7)
        shift.pack(side="left", padx=4)
        self.controls.append((shift, "normal"))
        shift.bind("<Return>", lambda event: self.apply_telemetry_shift())
        apply_shift = ttk.Button(options, text="Apply", command=self.apply_telemetry_shift)
        apply_shift.pack(side="left")
        self.controls.append((apply_shift, "normal"))
        ttk.Label(options, text="Velocity comparison: ECEF").pack(side="left", padx=(10, 4))
        self.telemetry_info = tk.StringVar(value="No telemetry CSV loaded. Positive time shift moves telemetry later.")
        ttk.Label(case_panel, textvariable=self.telemetry_info, wraplength=850).grid(
            row=5, column=0, columnspan=4, sticky="w", padx=4)
        self.notebook = ttk.Notebook(left)
        self.notebook.grid(row=3, column=0, sticky="nsew")
        for phase in RESULT_TABS:
            tab = ttk.Frame(self.notebook)
            figure = Figure(figsize=(10, 6), dpi=100, constrained_layout=True)
            canvas = FigureCanvasTkAgg(figure, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.tabs[phase], self.figures[phase], self.canvases[phase] = tab, figure, canvas
            self.notebook.add(tab, text=phase)
        self.notebook.hide(self.tabs["Booster"])
        fit_tab = ttk.Frame(self.notebook)
        self.tabs["Fit results"] = fit_tab
        self.notebook.add(fit_tab, text="Fit results")
        self.fit_report_view = ScrolledText(fit_tab, wrap="word", state="disabled", font=("TkFixedFont", 10))
        self.fit_report_view.pack(fill="both", expand=True, padx=10, pady=10)
        self.clear_plots("Run optimization to display this phase.")
        logs = ttk.LabelFrame(left, text="Solver output", padding=4)
        logs.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.log = ScrolledText(logs, height=7, wrap="word", state="disabled", font=("TkFixedFont", 9))
        self.log.pack(fill="both", expand=True)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self.settings_title = tk.StringVar(value="Case 1 settings")
        ttk.Label(right, textvariable=self.settings_title, style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=12)
        settings = ttk.Frame(right, padding=12)
        settings.grid(row=1, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)
        self.vehicle = tk.StringVar(value="Falcon 9")
        self.recovery = tk.StringVar(value=next(iter(RECOVERY)))
        for row, (label, variable, choices) in enumerate((("Vehicle", self.vehicle, self.vehicles), ("Recovery", self.recovery, RECOVERY))):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            combo = ttk.Combobox(settings, textvariable=variable, values=list(choices), state="readonly", width=28)
            combo.grid(row=row, column=1, sticky="ew", pady=4)
            if row == 0:
                self.vehicle_combo = combo
            self.controls.append((combo, "readonly"))
        self.new_vehicle_button = ttk.Button(settings, text="New vehicle…", command=self.open_vehicle_dialog)
        self.new_vehicle_button.grid(row=0, column=2, sticky="e", padx=(6, 0))
        self.controls.append((self.new_vehicle_button, "normal"))
        orbit_form = ttk.LabelFrame(settings, text="Target orbit", padding=6)
        orbit_form.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        orbit_form.columnconfigure(1, weight=1)
        self.orbit_variables = {}
        for row, (name, label, default) in enumerate((
                ("apogee", "Apogee altitude [km]", 200.0),
                ("perigee", "Perigee altitude [km]", 200.0),
                ("inclination", "Inclination [deg]", 28.6))):
            ttk.Label(orbit_form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            variable = tk.StringVar(value=str(default))
            entry = ttk.Entry(orbit_form, textvariable=variable, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            self.orbit_variables[name] = variable
            self.controls.append((entry, "normal"))
            variable.trace_add("write", self.inputs_changed)
        ttk.Label(settings, text="Payload mass [kg]").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.payload_mass = tk.StringVar(value="-1")
        payload_entry = ttk.Entry(settings, textvariable=self.payload_mass, width=12)
        payload_entry.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        self.controls.append((payload_entry, "normal"))
        self.payload_mass.trace_add("write", self.inputs_changed)
        ttk.Label(settings, text="Positive: fixed payload · 0 or negative: maximize").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        site_form = ttk.LabelFrame(settings, text="Launch site", padding=6)
        site_form.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        site_form.columnconfigure(1, weight=1)
        self.launch_site = tk.StringVar(value=DEFAULT_LAUNCH_SITE)
        site_combo = ttk.Combobox(site_form, textvariable=self.launch_site,
                                  values=list(LAUNCH_SITES), state="readonly", width=26)
        site_combo.grid(row=0, column=0, columnspan=2, sticky="ew")
        site_combo.bind("<<ComboboxSelected>>", self.choose_launch_site)
        self.controls.append((site_combo, "readonly"))
        self.site_variables = {}
        for row, (key, label) in enumerate((("latitude", "Latitude [deg N]"),
                                            ("longitude", "Longitude [deg E]"),
                                            ("altitude", "Altitude [m]")), start=1):
            ttk.Label(site_form, text=label).grid(row=row, column=0, sticky="w", pady=2)
            variable = tk.StringVar(value=launch_site_inputs()[key])
            entry = ttk.Entry(site_form, textvariable=variable, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.site_variables[key] = variable
            self.controls.append((entry, "normal"))
            variable.trace_add("write", self.site_coordinates_changed)
        ttk.Label(site_form, text="Approximate presets · altitude + LaunchAltDelta").grid(
            row=4, column=0, columnspan=2, sticky="w")
        form = ttk.LabelFrame(right, text="DispesrionFactorsType", padding=6)
        form.grid(row=2, column=0, sticky="nsew", padx=(12, 0))
        form.columnconfigure(0, weight=1)
        form.rowconfigure(0, weight=1)
        scroll = tk.Canvas(form, highlightthickness=0, width=365)
        bar = ttk.Scrollbar(form, orient="vertical", command=scroll.yview)
        scroll.configure(yscrollcommand=bar.set)
        scroll.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        entries = ttk.Frame(scroll)
        window = scroll.create_window((0, 0), window=entries, anchor="nw")
        entries.bind("<Configure>", lambda event: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda event: scroll.itemconfigure(window, width=event.width))
        entries.columnconfigure(1, weight=1)
        defaults = asdict(DispesrionFactorsType())
        for row, (name, default) in enumerate(defaults.items()):
            label = ttk.Label(entries, wraplength=235, justify="left")
            label.grid(row=row, column=0, sticky="w", padx=4, pady=8)
            self.dispersion_labels[name] = label
            variable = tk.StringVar(value=str(default))
            entry = ttk.Entry(entries, textvariable=variable, width=10)
            entry.grid(row=row, column=1, sticky="ew", padx=4, pady=8)
            ttk.Label(entries, text=field_unit(name)).grid(row=row, column=2, sticky="w", padx=4)
            self.variables[name] = variable
            self.controls.append((entry, "normal"))
            variable.trace_add("write", self.inputs_changed)
        self.vehicle.trace_add("write", self.inputs_changed)
        self.vehicle.trace_add("write", self.refresh_dispersion_nominals)
        self.site_variables['altitude'].trace_add("write", self.refresh_dispersion_nominals)
        self.refresh_dispersion_nominals()
        self.recovery.trace_add("write", self.inputs_changed)

        footer = ttk.Frame(right, padding=(12, 10, 0, 0))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        footer.columnconfigure(1, weight=1)
        self.save_state_button = ttk.Button(footer, text="Save state…", command=self.save_state_file)
        self.save_state_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.load_state_button = ttk.Button(footer, text="Load state…", command=self.load_state_file)
        self.load_state_button.grid(row=0, column=1, sticky="ew")
        reset = ttk.Button(footer, text="Reset dispersion to defaults", command=self.reset_defaults)
        reset.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(6, 0))
        self.reset_gui_button = ttk.Button(footer, text="Reset GUI", command=self.reset_gui)
        self.reset_gui_button.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.controls.append((self.save_state_button, "normal"))
        self.controls.append((self.load_state_button, "normal"))
        self.controls.append((self.reset_gui_button, "normal"))
        self.controls.append((reset, "normal"))
        self.status = tk.StringVar(value="Ready — nominal dispersion values loaded.")
        ttk.Label(footer, textvariable=self.status, wraplength=360).grid(row=2, column=0, columnspan=2, sticky="w", pady=10)
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.cancel_button = ttk.Button(footer, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.grid(row=4, column=0, sticky="w", padx=(0, 8))
        self.run_button = ttk.Button(footer, text="Run Case 1", style="Run.TButton", command=self.run)
        self.run_button.grid(row=4, column=1, sticky="e")
        self.fit_button = ttk.Button(footer, text="Fit dispersion to telemetry…", command=self.open_fit_dialog, state="disabled")
        self.fit_button.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.restore_state()
        self.update_case_labels()
        self.restoring_state = False
        self.notebook.bind("<<NotebookTabChanged>>", self.schedule_save)
        self.panes.bind("<ButtonRelease-1>", self.schedule_save)
        root.bind("<Configure>", self.window_configured)
        self.schedule_layout_restore()
        for variable in (*self.case_visible, self.telemetry_visible,
                         self.telemetry_primary, self.telemetry_shift_input):
            variable.trace_add("write", self.schedule_save)

    def window_configured(self, event):
        if event.widget == self.root:
            self.schedule_layout_restore()
            self.schedule_save()

    def schedule_layout_restore(self):
        if self.pending_sash is None or self.closing:
            return
        if self.layout_id is not None:
            self.root.after_cancel(self.layout_id)
        # The window manager can resize a newly mapped window after Tk's idle
        # layout. Apply the saved divider once those configure events settle.
        self.layout_id = self.root.after(100, self.restore_divider)

    def restore_divider(self):
        self.layout_id = None
        self.panes.sashpos(0, min(self.pending_sash, max(1, self.panes.winfo_width() - 380)))
        self.pending_sash = None

    def schedule_save(self, *args):
        if self.restoring_state or self.closing:
            return
        if self.save_id is not None:
            self.root.after_cancel(self.save_id)
        self.save_id = self.root.after(500, self.save_state)

    def state_snapshot(self):
        """Capture every restorable GUI setting and completed result."""
        self.cases[self.selected_case].draft = self.capture_inputs()
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        cases = []
        for case, visible in zip(self.cases, self.case_visible):
            state = case.state
            if state.startswith(("Computing", "Fitting")):
                state = "Interrupted; previous result retained" if case.result is not None else "Interrupted"
            cases.append({"draft": case.draft, "color": case.color, "visible": visible.get(),
                          "computed_inputs": case.computed_inputs, "request": case.request,
                          "result": case.result, "run_dir": str(case.run_dir) if case.run_dir else None,
                          "state": state, "fit_report": case.fit_report, "fit_settings": case.fit_settings})
        return {"cases": cases, "selected_case": self.selected_case,
                "custom_vehicles": [value for value in self.vehicles.values() if isinstance(value, dict)],
                "telemetry": {"path": str(self.telemetry.path.resolve()) if self.telemetry else None,
                              "color": self.telemetry_color, "visible": self.telemetry_visible.get(),
                              "primary": self.telemetry_primary.get(), "shift": self.telemetry_shift,
                              "shift_input": self.telemetry_shift_input.get()},
                "layout": {"geometry": self.root.geometry(), "tab": selected_tab,
                           "zoomed": self.window_zoomed(), "sash": self.panes.sashpos(0)}}

    def save_state(self, path=None):
        if self.save_id is not None:
            self.root.after_cancel(self.save_id)
            self.save_id = None
        target = Path(path) if path is not None else self.state_path
        try:
            write_gui_state(target, self.state_snapshot())
        except (OSError, ValueError, TypeError) as exc:
            self.status.set(f"Could not save GUI settings: {exc}")
            self.append_log(f"\nCould not save GUI settings to {target}: {exc}\n")
            return False
        return True

    def save_state_file(self):
        from tkinter import filedialog
        if self.process is not None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save GUI state", defaultextension=".json",
            initialdir=str(self.output_root), initialfile="launch_optimization_state.json",
            filetypes=(("GUI state", "*.json"), ("All files", "*.*")))
        if not path:
            return
        if self.save_state(path):
            # Keep automatic startup restore synchronized with the exported file.
            self.save_state()
            self.status.set(f"GUI state saved to {path}")

    def load_state_file(self):
        from tkinter import filedialog, messagebox
        if self.process is not None:
            return
        # Flush the exact current state so a failed import can be rolled back.
        if not self.save_state():
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Load GUI state", initialdir=str(self.output_root),
            filetypes=(("GUI state", "*.json"), ("All files", "*.*")))
        if not path:
            return
        if not self.restore_state(path, reset_on_error=False):
            error = self.status.get()
            self.restore_state()
            self.status.set(error)
            messagebox.showerror("Could not load GUI state", error, parent=self.root)
            return
        self.save_state()
        self.status.set(f"GUI state loaded from {path}")

    def window_zoomed(self, value=None):
        import tkinter as tk
        try:
            if self.root.tk.call("tk", "windowingsystem") == "win32":
                if value is not None:
                    self.root.state("zoomed" if value else "normal")
                return self.root.state() == "zoomed"
            if value is not None:
                self.root.attributes("-zoomed", value)
            return bool(self.root.attributes("-zoomed"))
        except tk.TclError:
            return False  # Some window managers do not support maximization.

    def restore_state(self, path=None, reset_on_error=True):
        source_path = Path(path) if path is not None else self.state_path
        if not source_path.exists():
            return False
        notes = []
        restoring = self.restoring_state
        self.restoring_state = True
        try:
            state = read_gui_state(source_path)
            self.vehicles = vehicle_catalog(state.get('custom_vehicles', []))
            self.vehicle_combo.configure(values=list(self.vehicles))
            for index, saved in enumerate(state["cases"]):
                case = self.cases[index]
                draft = default_case_inputs()
                source = saved.get("draft", {})
                for key, options in (("vehicle", self.vehicles), ("recovery", RECOVERY)):
                    if source.get(key) in options:
                        draft[key] = source[key]
                if isinstance(source.get("payload_mass"), str):
                    draft["payload_mass"] = source["payload_mass"]
                for group in ("orbit", "dispersion", "launch_site"):
                    for key in draft[group]:
                        value = source.get(group, {}).get(key)
                        if isinstance(value, str):
                            draft[group][key] = value
                if draft["launch_site"]["name"] not in LAUNCH_SITES:
                    draft["launch_site"]["name"] = "Custom"
                case.draft = draft
                color = saved.get("color", case.color)
                self.root.winfo_rgb(color)
                case.color = color
                self.color_buttons[index].configure(background=color, activebackground=color)
                self.case_visible[index].set(bool(saved.get("visible", True)))
                case.result = saved.get("result")
                case.request = saved.get("request")
                case.computed_inputs = saved.get("computed_inputs")
                if case.computed_inputs is not None:
                    case.computed_inputs.setdefault("launch_site", launch_site_inputs())
                case.run_dir = Path(saved["run_dir"]) if saved.get("run_dir") else None
                case.state = saved.get("state", "Complete" if case.result is not None else "Not run")
                case.fit_report = saved.get("fit_report")
                case.fit_settings = saved.get("fit_settings")
            # Populate the editor before selecting another case, preserving drafts.
            self.load_case_form(0)
            index = state.get("selected_case", 0)
            self.select_case(index if type(index) is int and 0 <= index < 3 else 0)
            telemetry = state.get("telemetry", {})
            self.telemetry = None
            self.telemetry_color = telemetry.get("color", "#CC79A7")
            self.root.winfo_rgb(self.telemetry_color)
            self.telemetry_color_button.configure(background=self.telemetry_color)
            self.telemetry_visible.set(bool(telemetry.get("visible", True)))
            primary = telemetry.get("primary", "Stage 1 / booster")
            if primary in PRIMARY_GROUPS:
                self.telemetry_primary.set(primary)
            shift = float(telemetry.get("shift", 0))
            self.telemetry_shift = shift if np.isfinite(shift) else 0.0
            self.telemetry_shift_input.set(telemetry.get("shift_input", str(self.telemetry_shift)))
            if telemetry.get("path"):
                try:
                    self.telemetry = load_telemetry_csv(telemetry["path"])
                    self.select_telemetry_column_mode()
                except (OSError, ValueError, UnicodeError) as exc:
                    notes.append(f"Telemetry could not be reopened: {exc}. Use Load telemetry CSV to select it again.")
            self.refresh_comparison()
            layout = state.get("layout", {})
            if isinstance(layout.get("geometry"), str):
                self.root.geometry(layout["geometry"])
            if isinstance(layout.get("zoomed"), bool):
                self.window_zoomed(layout["zoomed"])
            # Layout must settle before restoring the divider position.
            self.root.update_idletasks()
            if isinstance(layout.get("sash"), int) and layout["sash"] > 0:
                self.pending_sash = layout["sash"]
                self.panes.sashpos(0, min(layout["sash"], max(1, self.panes.winfo_width() - 380)))
            tab = layout.get("tab", "First stage")
            if tab in self.tabs and self.notebook.tab(self.tabs[tab], "state") != "hidden":
                self.notebook.select(self.tabs[tab])
            self.status.set("GUI settings and completed cases restored." if not notes else notes[0])
        except Exception as exc:
            if reset_on_error:
                # A damaged or obsolete startup snapshot must not prevent launch.
                self.reset_gui(save=False)
                notes.append(f"Could not restore GUI settings; defaults loaded: {exc}")
            else:
                notes.append(f"Could not load GUI state: {exc}")
            self.status.set(notes[-1])
            return False
        finally:
            self.restoring_state = restoring
        for note in notes:
            self.append_log(note + "\n")
        return True

    def reset_gui(self, save=True):
        if self.process is not None:
            return
        if self.layout_id is not None:
            self.root.after_cancel(self.layout_id)
            self.layout_id = None
        self.pending_sash = None
        restoring = self.restoring_state
        self.restoring_state = True
        try:
            self.cases = [ComparisonCase(f"Case {i + 1}", color) for i, color in
                          enumerate(("#0072B2", "#D55E00", "#009E73"))]
            self.load_case_form(0)
            for index, case in enumerate(self.cases):
                self.case_visible[index].set(True)
                self.color_buttons[index].configure(background=case.color, activebackground=case.color)
            self.telemetry = None
            self.telemetry_color = "#CC79A7"
            self.telemetry_color_button.configure(background=self.telemetry_color)
            self.telemetry_visible.set(True)
            self.telemetry_primary.set("Stage 1 / booster")
            self.telemetry_shift = 0.0
            self.telemetry_shift_input.set("0")
            self.request = self.result = self.run_dir = self.submitted_inputs = None
            self.refresh_comparison()
            self.notebook.select(self.tabs["First stage"])
            self.window_zoomed(False)
            self.root.geometry("1440x900")
            self.root.update_idletasks()
            self.panes.sashpos(0, max(1, self.panes.winfo_width() - 420))
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.configure(state="disabled")
            self.status.set("Default GUI restored. Saved run files remain on disk.")
        finally:
            self.restoring_state = restoring
        if save:
            self.save_state()

    def choose_launch_site(self, event=None):
        if self.process is not None:
            return
        if self.launch_site.get() != "Custom":
            self.loading_case = True
            try:
                for key, value in launch_site_inputs(self.launch_site.get()).items():
                    if key != "name":
                        self.site_variables[key].set(value)
            finally:
                self.loading_case = False
        self.inputs_changed()

    def site_coordinates_changed(self, *args):
        if not self.loading_case:
            self.launch_site.set("Custom")
            self.inputs_changed()

    def capture_inputs(self):
        return {"vehicle": self.vehicle.get(), "recovery": self.recovery.get(),
                "payload_mass": self.payload_mass.get(),
                "launch_site": dict(name=self.launch_site.get(), **{k: v.get() for k, v in self.site_variables.items()}),
                "dispersion": {k: v.get() for k, v in self.variables.items()},
                "orbit": {k: v.get() for k, v in self.orbit_variables.items()}}

    def refresh_dispersion_nominals(self, *args):
        values = nominal_dispersion_values(self.vehicles[self.vehicle.get()], self.site_variables['altitude'].get())
        for name, label in self.dispersion_labels.items():
            label.configure(text=f"{name} ({values[name]})")

    def inputs_changed(self, *args):
        if self.loading_case or not hasattr(self, "status"):
            return
        if self.process is None:
            self.cases[self.selected_case].draft = self.capture_inputs()
            self.update_case_labels()
            self.status.set(f"{self.cases[self.selected_case].name}: inputs changed — run to apply.")
            self.schedule_save()

    def select_case(self, index):
        if self.process is not None:
            self.case_selection.set(self.selected_case)
            return
        self.cases[self.selected_case].draft = self.capture_inputs()
        self.load_case_form(index)

    def load_case_form(self, index):
        self.selected_case = index
        self.case_selection.set(index)
        case = self.cases[index]
        self.loading_case = True
        try:
            self.vehicle.set(case.draft["vehicle"])
            self.recovery.set(case.draft["recovery"])
            self.payload_mass.set(case.draft["payload_mass"])
            site = case.draft.get("launch_site", launch_site_inputs())
            self.launch_site.set(site["name"])
            for key, variable in self.site_variables.items():
                variable.set(site[key])
            for key, value in case.draft["dispersion"].items():
                self.variables[key].set(value)
            for key, value in case.draft["orbit"].items():
                self.orbit_variables[key].set(value)
        finally:
            self.loading_case = False
        self.result = case.result
        self.settings_title.set(f"{case.name} settings")
        self.update_case_labels()
        self.status.set(f"Editing {case.name}. Run to apply its current settings.")
        self.schedule_save()

    def choose_color(self, index):
        from tkinter.colorchooser import askcolor
        color = askcolor(color=self.cases[index].color, title=f"{self.cases[index].name} color", parent=self.root)[1]
        if color:
            self.cases[index].color = color
            self.color_buttons[index].configure(background=color, activebackground=color)
            self.refresh_comparison()

    def update_case_labels(self):
        for index, case in enumerate(self.cases):
            text = case.state
            if case.result is not None:
                text += (f" · Payload: {float(case.result['payload_mass']):,.1f} kg"
                         f" · S2 propellant left (est.): {remaining_propellant_text(case)}")
                if case.edited:
                    text += " · edited inputs; showing previous result"
            self.case_status[index].set(text)
        case = self.cases[self.selected_case]
        self.run_button.configure(text=f"{'Recompute' if case.result is not None else 'Run'} {case.name}")
        can_sync = self.process is None and self.telemetry is not None and case.result is not None
        self.sync_button.configure(text=f"Sync to {case.name}", state="normal" if can_sync else "disabled")
        self.fit_button.configure(state="normal" if self.process is None and self.telemetry is not None else "disabled")
        self.render_fit_report()

    def load_telemetry(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(parent=self.root, title="Select telemetry CSV",
                                          initialdir=PROJECT_DIR / "falcon9_telemetry",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if not path:
            return
        try:
            data = load_telemetry_csv(path)
        except (ValueError, OSError, UnicodeError) as exc:
            messagebox.showerror("Could not load telemetry", str(exc), parent=self.root)
            return
        self.telemetry = data
        self.select_telemetry_column_mode()
        # Pick a distinct default even if a simulation case was recolored.
        if self.telemetry_color.lower() in {case.color.lower() for case in self.cases}:
            self.telemetry_color = next(color for color in ("#CC79A7", "#7B2CBF", "#222222", "#E69F00")
                                        if color.lower() not in {case.color.lower() for case in self.cases})
            self.telemetry_color_button.configure(background=self.telemetry_color)
        self.telemetry_shift = 0.0
        self.telemetry_shift_input.set("0")
        self.telemetry_visible.set(True)
        self.refresh_comparison()

    def select_telemetry_column_mode(self):
        """Numbered/prefixed phase columns need no legacy primary assignment."""
        if not self.telemetry.groups['primary']:
            self.telemetry_primary.set('Phase columns')
        elif self.telemetry_primary.get() == 'Phase columns':
            self.telemetry_primary.set('Stage 1 / booster')

    def clear_telemetry(self):
        self.telemetry = None
        self.telemetry_shift = 0.0
        self.telemetry_shift_input.set("0")
        self.refresh_comparison()

    def choose_telemetry_color(self):
        from tkinter.colorchooser import askcolor
        color = askcolor(color=self.telemetry_color, title="Telemetry color", parent=self.root)[1]
        if color:
            self.telemetry_color = color
            self.telemetry_color_button.configure(background=color)
            self.refresh_comparison()

    def sync_telemetry(self):
        from tkinter import messagebox
        case = self.cases[self.selected_case]
        if self.process is not None or self.telemetry is None or case.result is None:
            return
        try:
            channels = self.telemetry.channels("stage1", self.telemetry_primary.get())
            if "speed" not in channels:
                raise ValueError("No first-stage speed telemetry is available. Load velocity1_kmh "
                                 "or select Stage 1 / booster for legacy unnumbered speed columns.")
            initial_time = float(np.asarray(case.result["t1_vec"]).ravel()[0])
            initial_local = np.asarray(case.result["x1"])[2:4, 0]
            initial_velocity = local_velocity_in_ecef(case.result, [initial_local[0], 0., initial_local[1]])
            initial_speed = float(np.linalg.norm(initial_velocity))
            if not np.isfinite(initial_time):
                raise ValueError("The selected case has an invalid initial time.")
            matched_time = interpolate_speed_time(self.telemetry.time, channels["speed"], initial_speed)
            shift = initial_time - matched_time
        except ValueError as exc:
            messagebox.showerror("Could not sync telemetry", str(exc), parent=self.root)
            return
        # Recalculate from original CSV times so repeated clicks never accumulate shifts.
        self.telemetry_shift = shift
        self.telemetry_shift_input.set(str(shift))
        self.telemetry_visible.set(True)
        self.refresh_comparison()
        self.status.set(f"Synced to {case.name}'s computed result: {initial_speed:.3f} m/s at "
                        f"telemetry {matched_time:.3f} s → case {initial_time:.3f} s "
                        f"(shift {shift:+.3f} s).")

    def apply_telemetry_shift(self):
        from tkinter import messagebox
        try:
            shift = float(self.telemetry_shift_input.get())
            if not np.isfinite(shift):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid telemetry time shift", "Enter a finite shift in seconds.", parent=self.root)
            return
        self.telemetry_shift = shift
        self.refresh_comparison()

    def refresh_comparison(self):
        displayed = [case for case, visible in zip(self.cases, self.case_visible)
                     if visible.get() and case.result is not None]
        has_booster = any("x4" in case.result for case in displayed)
        telemetry = self.telemetry if self.telemetry_visible.get() else None
        primary = self.telemetry_primary.get()
        if telemetry and (telemetry.groups["booster"] or primary == "Booster" and telemetry.groups["primary"]):
            has_booster = True
        if has_booster:
            self.notebook.add(self.tabs["Booster"], text="Booster")
        else:
            if self.notebook.select() == str(self.tabs["Booster"]):
                self.notebook.select(self.tabs["First stage"])
            self.notebook.hide(self.tabs["Booster"])
        matched_tabs = 0
        for phase, figure in self.figures.items():
            if phase == '3D trajectory':
                draw_trajectory_3d(figure, [(case.result, case.request['target_orbit'], case.color, case.name)
                                          for case in displayed])
                self.canvases[phase].draw_idle()
                continue
            channels = telemetry.channels(phase_group(phase), primary) if telemetry else {}
            speed_frame = "ECEF" if any(key in channels for key in ("speed", "vx", "vy", "vz")) else "Inertial"
            phase_cases = [case for case in displayed if phase != "Booster" or "x4" in case.result]
            if not phase_cases:
                figure.clear()
            for index, case in enumerate(phase_cases):
                draw_phase(figure, phase, case.result, case.request["target_orbit"],
                           color=case.color, case_label=case.name, append=index > 0, speed_frame=speed_frame)
            windows = []
            # Only shared, unlabelled ascent/return readings need a simulation
            # window for routing. Explicit phase telemetry must show its full
            # extent, including a landing later than the simulated landing.
            shared_primary = (telemetry is not None and primary == "Stage 1 / booster"
                              and bool(telemetry.groups["primary"]))
            if shared_primary:
                for case in phase_cases:
                    if phase_group(phase) == "stage1":
                        windows.append((min(0, float(np.min(case.result["t1_vec"]))), float(np.max(case.result["t1_vec"]))))
                    elif phase == "Booster":
                        windows.append((float(np.min(case.result["t4_vec"])), float(np.max(case.result["t4_vec"]))))
            show_telemetry = telemetry is not None and (phase != "Booster" or has_booster)
            overlaid = show_telemetry and overlay_telemetry(figure, phase, telemetry, primary,
                                                          self.telemetry_color, self.telemetry_shift, windows)
            if overlaid:
                matched_tabs += 1
                if phase == "Second stage" and "speed" in channels:
                    figure.axes[1].set_title("ECEF speed")
            if not phase_cases and not overlaid:
                figure.text(0.5, 0.5, "No displayed case or matching telemetry for this phase.",
                            ha="center", va="center", color="#666666")
            self.canvases[phase].draw_idle()
        if self.telemetry:
            count = sum(len(channels) for channels in self.telemetry.groups.values())
            self.telemetry_info.set(f"{self.telemetry.path.name} · {count} channels · {self.telemetry.time_note} · "
                                    f"shift {self.telemetry_shift:+g} s · "
                                    f"{str(matched_tabs) + ' tabs overlaid' if telemetry else 'hidden'}")
        else:
            self.telemetry_info.set("No telemetry CSV loaded. Positive time shift moves telemetry later.")
        summaries = []
        for case in displayed:
            orbit = case.request["target_orbit"]
            vehicle = vehicle_name(case.request["configuration"])
            site = parse_launch_site(case.request.get("launch_site"))
            summaries.append(f"{case.name} · {vehicle} · {site['name']} · {case.request['recovery']} · "
                             f"{(orbit['apogee'] - 6378137) / 1000:g} × "
                             f"{(orbit['perigee'] - 6378137) / 1000:g} km · "
                             f"{np.rad2deg(orbit['i']):g}° · Payload: {float(case.result['payload_mass']):,.1f} kg "
                             f"({'fixed' if case.request.get('payload_mass_predefined', -1) > 0 else 'maximized'}) · "
                             f"S2 propellant left (est.): {remaining_propellant_text(case)}")
        self.summary.set("\n".join(summaries) if summaries else
                         f"Telemetry only · {telemetry.path.name}" if telemetry and matched_tabs else
                         "No cases displayed. Compute a case and enable Display.")
        self.update_case_labels()
        self.schedule_save()

    def reset_defaults(self):
        for name, value in asdict(DispesrionFactorsType()).items():
            self.variables[name].set(str(value))
        self.status.set("Nominal dispersion values restored. Run to apply.")

    def clear_plots(self, message):
        for name, figure in self.figures.items():
            figure.clear()
            figure.text(0.5, 0.5, message, ha="center", va="center", color="#666666")
            self.canvases[name].draw_idle()

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        # Keep rendering cheap during very long solves; the full log stays on disk.
        count = int(self.log.index("end-1c").split(".")[0])
        if count > 1500:
            self.log.delete("1.0", f"{count - 1500}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_running(self, running):
        for control, idle_state in self.controls:
            control.configure(state="disabled" if running else idle_state)
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if running and self.job_kind == "fit":
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=self.fit_options["max_evaluations"], value=0)
        elif running:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
        self.update_case_labels()

    def run(self):
        from tkinter import messagebox
        if self.process is not None:
            return
        try:
            request = nominal_request({k: v.get() for k, v in self.variables.items()},
                                      self.vehicles[self.vehicle.get()], RECOVERY[self.recovery.get()],
                                      {k: v.get() for k, v in self.orbit_variables.items()},
                                      payload_mass=self.payload_mass.get(),
                                      launch_site=self.capture_inputs()["launch_site"])
        except ValueError as exc:
            self.status.set(str(exc))
            messagebox.showerror("Invalid run settings", str(exc), parent=self.root)
            return
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            case = self.cases[self.selected_case]
            prefix = f"case{self.selected_case + 1}_" + datetime.now().strftime("%Y%m%d_%H%M%S_")
            self.run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.output_root))
            (self.run_dir / "inputs.json").write_text(json.dumps(request, indent=2))
            (self.run_dir / "case.json").write_text(json.dumps({"name": case.name, "color": case.color}, indent=2))
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            env["PYTHONUNBUFFERED"] = "1"
            with (self.run_dir / "solver.log").open("wb") as output:
                self.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(self.run_dir)],
                    stdout=output, stderr=subprocess.STDOUT, cwd=PROJECT_DIR, env=env,
                )
            self.log_stream = (self.run_dir / "solver.log").open(errors="replace")
        except OSError as exc:
            if self.process is not None:
                self.process.kill()
                self.process.wait()
                self.process = None
            self.status.set(f"Could not start: {exc}")
            messagebox.showerror("Could not start optimization", str(exc), parent=self.root)
            return
        self.job_kind = "optimization"
        self.request, self.result = request, None
        self.running_case = self.selected_case
        self.submitted_inputs = self.capture_inputs()
        case.draft = deepcopy(self.submitted_inputs)
        case.state = "Computing (previous result retained)" if case.result is not None else "Computing"
        self.cancelled = False
        self.started = time.monotonic()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.append_log(f"{case.name} · Run files: {self.run_dir}\n")
        self.refresh_comparison()
        self.set_running(True)
        self.poll()

    def poll(self):
        if self.log_stream:
            self.append_log(self.log_stream.read(65536))
        code = self.process.poll()
        if code is None:
            if self.job_kind == "fit":
                self.poll_fit_progress()
            else:
                self.status.set(f"{'Cancelling' if self.cancelled else 'Optimizing'} · {time.monotonic() - self.started:.0f} s elapsed")
            self.poll_id = self.root.after(200, self.poll)
            return
        self.poll_id = None
        if self.log_stream:
            self.append_log(self.log_stream.read())
            self.log_stream.close()
            self.log_stream = None
        self.process = None
        if self.kill_id is not None:
            self.root.after_cancel(self.kill_id)
            self.kill_id = None
        case = self.cases[self.running_case]
        self.running_case = None
        self.set_running(False)
        if self.closing:
            self.destroy_window()
            return
        if self.job_kind == "fit":
            self.finish_fit(case, code)
            self.job_kind = "optimization"
            return
        if self.cancelled:
            case.state = "Cancelled; previous result retained" if case.result is not None else "Cancelled"
            self.status.set(f"{case.name}: {case.state}. You can run again.")
            self.refresh_comparison()
            return
        try:
            if code != 0:
                raise RuntimeError(f"Solver process exited with code {code}. See solver output.")
            path = self.run_dir / "solution.pickle"
            if not path.exists():
                path = self.run_dir / "solution_Failed.pickle"
            # Only read the result produced by our own worker in this run directory.
            with path.open("rb") as stream:
                result = pickle.load(stream)
            if not result.get("success"):
                raise RuntimeError(f"Optimization did not converge: {result.get('return_status', 'unknown status')}.")
            self.result = result
            case.result = result
            case.request = deepcopy(self.request)
            case.computed_inputs = deepcopy(self.submitted_inputs)
            case.run_dir = self.run_dir
            case.state = "Complete"
            case.fit_report = None
            self.refresh_comparison()
            self.status.set(f"{case.name} complete · {result.get('iter_count', '—')} iterations. Results and inputs saved.")
            self.append_log(f"\nSolution saved: {path}\n")
        except Exception as exc:
            case.state = "Failed; previous result retained" if case.result is not None else "Failed"
            self.status.set(f"{case.name}: {exc}")
            self.refresh_comparison()
            self.append_log(f"\n{exc}\nRun files: {self.run_dir}\n")

    def render_fit_report(self):
        case = self.cases[self.selected_case]
        report = case.fit_report
        if not report:
            text = (f"{case.name}: no telemetry fit yet.\n\n"
                    "Load telemetry and click Fit dispersion to telemetry. Choose the factors, bounds, "
                    "evaluation budget, and destination case. Every candidate is synchronized before comparison.")
        else:
            before, best = report["baseline"], report["best"]
            lines = [f"{case.name} — telemetry fitting", str(report["termination"]),
                     f"Evaluations: {report['evaluations']}",
                     f"Fixed payload: {best['request']['payload_mass_predefined']:,.1f} kg",
                     f"Score: {before['score']:.6g} → {best['score']:.6g} (lower is better)",
                     f"Synchronization: {best['synchronization']['phase']} · telemetry shift "
                     f"{best['synchronization']['shift']:+.6g} s", "", "Dispersion factors", ""]
            for name in report['parameters']:
                a, b = before['request']['dispersion'][name], best['request']['dispersion'][name]
                low, high = report['bounds'][name]
                lines.append(f"{name:26s} {a:12.6g} → {b:12.6g}   bounds [{low:g}, {high:g}]")
            if report.get('settings'):
                lines += ["", "Error scales (smaller means a stronger penalty)"]
                for channel, unit in CHANNEL_UNITS.items():
                    key = f'{channel}_scale'
                    if key in report['settings']:
                        lines.append(f"{channel}: {float(report['settings'][key]):g} {unit}")
                if 'phase_time_scale' in report['settings']:
                    lines.append(f"Phase timing: {float(report['settings']['phase_time_scale']):g} s")
            lines += ["", "Telemetry errors (RMSE)", ""]
            for name, metric in best['errors'].items():
                a = before['errors'][name]['rmse']
                lines.append(f"{name:22s} {a:12.4g} → {metric['rmse']:12.4g} {metric['unit']} "
                             f"  coverage {100*metric['coverage']:.1f}%")
                if 'measured_samples' in metric:
                    lines.append(f"  {metric['measured_samples']} supplied / {metric['calculated_samples']} calculated samples")
            if best.get('phase_timing'):
                lines += ["", "Phase durations and late simulation samples", ""]
                for phase, metric in best['phase_timing'].items():
                    previous = before.get('phase_timing', {}).get(phase, metric)
                    lines.append(f"{phase}: telemetry {metric['telemetry_duration']:.6g} s; simulation "
                                 f"{previous['simulation_duration']:.6g} → {metric['simulation_duration']:.6g} s "
                                 f"(difference {metric['duration_error']:+.6g} s)")
                    lines.append(f"  {metric['late_samples']} late samples; "
                                 f"end overrun {metric['end_overrun']:.6g} s; timing penalty {metric['penalty']:.6g}")
            lines += ["", "The best candidate is displayed on the flight plots in this case's color.",
                      "Each comparison uses a newly interpolated telemetry synchronization.",
                      "", "Search history", ""]
            for trial in report.get('history', []):
                score = trial.get('score')
                score = f"{score:.6g}" if score is not None else "—"
                lines.append(f"{trial['evaluation']:4d}   score {score:>12s}   {trial['status']}")
            text = "\n".join(lines)
        if getattr(self, '_fit_report_text', None) != text:
            self.fit_report_view.configure(state="normal")
            self.fit_report_view.delete("1.0", "end")
            self.fit_report_view.insert("1.0", text)
            self.fit_report_view.configure(state="disabled")
            self._fit_report_text = text

    def form_request(self):
        draft = self.capture_inputs()
        return nominal_request(draft['dispersion'], self.vehicles[draft['vehicle']], RECOVERY[draft['recovery']],
                               draft['orbit'], draft['payload_mass'], draft['launch_site'])

    def add_vehicle(self, definition):
        definitions = [value for value in self.vehicles.values() if isinstance(value, dict)]
        catalog = vehicle_catalog([*definitions, definition])
        name = definition['name'].strip()
        self.vehicles = catalog
        self.vehicle_combo.configure(values=list(catalog))
        self.vehicle.set(name)
        self.status.set(f"Created {name}. Run the selected case to simulate it.")
        self.schedule_save()

    def open_vehicle_dialog(self):
        import tkinter as tk
        from tkinter import ttk, messagebox
        if self.process is not None:
            return
        selected = self.vehicle.get()
        configuration = self.vehicles[selected]
        defaults = vehicle_editor_values(configuration)
        model = VLType()
        model.ApplyVehicleConfiguration(configuration)
        dialog = tk.Toplevel(self.root)
        dialog.title(f"New vehicle from {selected}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry('760x640')
        dialog.minsize(620, 480)
        panel = ttk.Frame(dialog, padding=12)
        panel.pack(fill='both', expand=True)
        name_row = ttk.Frame(panel)
        name_row.pack(fill='x')
        ttk.Label(name_row, text='Vehicle name').pack(side='left', padx=(0, 8))
        name = f"{selected} copy"
        number = 2
        while name.casefold() in {key.casefold() for key in self.vehicles}:
            name = f"{selected} copy {number}"
            number += 1
        name_input = tk.StringVar(value=name)
        name_entry = ttk.Entry(name_row, textvariable=name_input)
        name_entry.pack(side='left', fill='x', expand=True)
        ttk.Label(panel, text=(f"Initial values: {selected}, before dispersion. "
                  "t = metric tons; tf = metric ton-force; s = seconds. "
                  "Launch site, target orbit, and payload use the selected case's settings."),
                  wraplength=710).pack(fill='x', pady=10)
        notebook = ttk.Notebook(panel)
        notebook.pack(fill='both', expand=True)
        variables = {}
        for group, parameters in VEHICLE_PARAMETER_GROUPS.items():
            tab = ttk.Frame(notebook)
            notebook.add(tab, text={'Ascent aerodynamics': 'Ascent aero', 'Booster recovery': 'Recovery'}.get(group, group))
            canvas = tk.Canvas(tab, highlightthickness=0)
            scrollbar = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            fields_panel = ttk.Frame(canvas, padding=8)
            window = canvas.create_window((0, 0), window=fields_panel, anchor='nw')
            fields_panel.bind('<Configure>', lambda event, c=canvas: c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>', lambda event, c=canvas, w=window: c.itemconfigure(w, width=event.width))
            fields_panel.columnconfigure(1, weight=1)
            for row, (key, label, unit, _) in enumerate(parameters):
                ttk.Label(fields_panel, text=label).grid(row=row, column=0, sticky='w', padx=4, pady=6)
                variable = tk.StringVar(value=defaults[key])
                ttk.Entry(fields_panel, textvariable=variable, width=20).grid(
                    row=row, column=1, sticky='ew', padx=8, pady=6)
                ttk.Label(fields_panel, text=unit).grid(row=row, column=2, sticky='w', padx=4)
                variables[key] = variable
        reference = ttk.Frame(notebook, padding=12)
        notebook.add(reference, text='Context')
        context = (
            f"Earth reference radius: {model.R0:,.3f} m\n"
            f"Earth gravitational parameter: {model.mu:.8g} m³/s²\n"
            f"Earth rotation: {model.omega_earth:.8g} rad/s\n"
            f"Earth eccentricity: {model.e:g}\n"
            f"Reference gravity: {model.g0:.6g} m/s²\n\n"
            "Reference area, total mass, stage masses, and propellant partition are calculated from your entries.\n\n"
            "The Earth atmosphere, flight equations, recovery engine fractions, and solver discretization use "
            "the existing simulation model. Launch coordinates, orbit, payload, and dispersion are set in the main window."
        )
        ttk.Label(reference, text=context, wraplength=680, justify='left').pack(anchor='nw')
        footer = ttk.Frame(panel)
        footer.pack(fill='x', pady=(10, 0))
        def create():
            try:
                definition = vehicle_from_editor(name_input.get(), {key: value.get() for key, value in variables.items()})
                self.add_vehicle(definition)
            except ValueError as exc:
                messagebox.showerror('Invalid vehicle', str(exc), parent=dialog)
                return
            dialog.destroy()
        ttk.Button(footer, text='Cancel', command=dialog.destroy).pack(side='left')
        ttk.Button(footer, text='Create vehicle', command=create).pack(side='right')
        name_entry.focus_set()

    def open_fit_dialog(self):
        import tkinter as tk
        from tkinter import ttk, messagebox
        if self.process is not None or self.telemetry is None:
            return
        try:
            request = self.form_request()
        except ValueError as exc:
            messagebox.showerror("Invalid case settings", str(exc), parent=self.root)
            return
        source = self.cases[self.selected_case]
        saved = source.fit_settings or {}
        bounds = default_fit_bounds(request)
        nominal_values = nominal_dispersion_values(request['configuration'], request['launch_site']['altitude'])
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Fit {source.name} to telemetry")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry('760x680')
        dialog.minsize(650, 480)
        footer = ttk.Frame(dialog, padding=12)
        footer.pack(side='bottom', fill='x')
        body = ttk.Frame(dialog)
        body.pack(fill='both', expand=True)
        canvas = tk.Canvas(body, highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        panel = ttk.Frame(canvas, padding=12)
        window = canvas.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
        dialog.bind('<MouseWheel>', lambda event: canvas.yview_scroll(-int(event.delta / 120), 'units'))
        dialog.bind('<Button-4>', lambda event: canvas.yview_scroll(-1, 'units'))
        dialog.bind('<Button-5>', lambda event: canvas.yview_scroll(1, 'units'))
        ttk.Label(panel, text="Choose factors and absolute bounds. Each candidate is synchronized before scoring.",
                  wraplength=630).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        for column, title in enumerate(("Fit factor", "Initial guess", "Lower bound", "Upper bound")):
            ttk.Label(panel, text=title).grid(row=1, column=column, sticky="w", padx=4)
        selected, lows, highs = {}, {}, {}
        for row, (name, pair) in enumerate(bounds.items(), start=2):
            selected[name] = tk.BooleanVar(value=name in saved.get('parameters', DEFAULT_PARAMETERS))
            lows[name], highs[name] = tk.StringVar(value=str(pair[0])), tk.StringVar(value=str(pair[1]))
            ttk.Checkbutton(panel, text=f"{name}\nNominal: {nominal_values[name]}",
                            variable=selected[name]).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            ttk.Label(panel, text=f"{request['dispersion'][name]:g}").grid(row=row, column=1, sticky="e", padx=8)
            ttk.Entry(panel, textvariable=lows[name], width=12).grid(row=row, column=2, padx=4)
            ttk.Entry(panel, textvariable=highs[name], width=12).grid(row=row, column=3, padx=4)
        row = 2 + len(bounds)
        settings = {}
        for key, label, default in (("max_evaluations", "Maximum solver evaluations", 100),
                                    ("altitude_scale", "Altitude error scale [m]", ERROR_SCALES['altitude_scale']),
                                    ("speed_scale", "Speed error scale [m/s]", ERROR_SCALES['speed_scale']),
                                    ("acceleration_scale", "Acceleration error scale [m/s²]", ERROR_SCALES['acceleration_scale']),
                                    ("pressure_scale", "Dynamic pressure error scale [kPa]", ERROR_SCALES['pressure_scale']),
                                    ("phase_time_scale", "Phase timing error scale [s]", ERROR_SCALES['phase_time_scale']),
                                    ("regularization", "Penalty for parameter changes", .01)):
            settings[key] = tk.StringVar(value=str(saved.get(key, default)))
            ttk.Label(panel, text=label).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
            ttk.Entry(panel, textvariable=settings[key], width=12).grid(row=row, column=2, columnspan=2, sticky="ew", padx=4)
            row += 1
        default_target = next((i for i, case in enumerate(self.cases)
                               if i != self.selected_case and case.result is None), self.selected_case)
        destination = tk.StringVar(value=self.cases[default_target].name)
        ttk.Label(panel, text="Save best result to case").grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Combobox(panel, textvariable=destination, values=[case.name for case in self.cases], state="readonly").grid(
            row=row, column=2, columnspan=2, sticky="ew", padx=4)
        row += 1
        ttk.Label(panel, wraplength=630, text=("The destination case receives the best fit when the search ends. "
                  "Bounds open at their nominal defaults; the initial guess uses the selected case's current values. "
                  "Nominal units: s = seconds, t = metric tons, tf = metric ton-force; SL = sea level, vac = vacuum. "
                  "Payload is held fixed; maximize mode uses the current result's payload or a preliminary nominal solve. "
                  "Smaller error scales give stronger mismatch penalties. Supplied acceleration is specific "
                  "acceleration; missing values use ECEF dv/dt. Pressure uses supplied values or ½ρv². "
                  "Acceleration and pressure penalties apply only to ascent. ASDS and RTLS also fit "
                  "booster return altitude/speed, including deceleration. "
                  "Missing telemetry rows are skipped. Phase durations must match telemetry; "
                  "simulation samples after the last phase reading receive an extra penalty. "
                  "Cancel retains the best completed comparison.")).grid(row=row, column=0, columnspan=4, sticky="w", pady=8)
        def remember(*args):
            source.fit_settings = dict(parameters=[name for name in selected if selected[name].get()],
                                       bounds={name: [lows[name].get(), highs[name].get()] for name in selected},
                                       **{key: variable.get() for key, variable in settings.items()})
            self.schedule_save()
        def start():
            remember()
            try:
                options = deepcopy(source.fit_settings)
                options['max_evaluations'] = int(options['max_evaluations'])
                for key in (*ERROR_SCALES, 'regularization'):
                    options[key] = float(options[key])
                options['bounds'] = {name: tuple(map(float, options['bounds'][name])) for name in options['parameters']}
                target = [case.name for case in self.cases].index(destination.get())
                self.start_fit(options, target)
            except (ValueError, OSError) as exc:
                messagebox.showerror("Could not start fitting", str(exc), parent=dialog)
                return
            dialog.destroy()
        for variable in (*selected.values(), *lows.values(), *highs.values(), *settings.values()):
            variable.trace_add('write', remember)
        ttk.Button(footer, text="Close", command=dialog.destroy).pack(side='left')
        ttk.Button(footer, text="Start search", command=start).pack(side='right')

    def start_fit(self, options, target):
        if self.process is not None:
            return
        if self.telemetry is None:
            raise ValueError('Load telemetry before fitting.')
        request = self.form_request()
        # Older sessions and callers may only contain altitude/speed scales.
        options = dict(ERROR_SCALES, **options)
        if not options['parameters'] or options['max_evaluations'] < 2:
            raise ValueError('Select factors and allow at least two evaluations.')
        scales = [options[key] for key in ERROR_SCALES]
        if not np.isfinite([*scales, options['regularization']]).all() or min(scales) <= 0 or options['regularization'] < 0:
            raise ValueError('Error scales must be positive and the parameter penalty nonnegative.')
        for name in options['parameters']:
            low, high = options['bounds'][name]
            if not np.isfinite([low, high]).all() or not low < high or not low <= request['dispersion'][name] <= high:
                raise ValueError(f'{name}: bounds must be finite, increasing, and contain the current value.')
            if name in MULTIPLIERS and low <= 0:
                raise ValueError(f'{name}: multiplier bounds must be positive.')
        if not any('speed' in self.telemetry.channels(phase, self.telemetry_primary.get()) for phase in ('stage1', 'stage2')):
            raise ValueError('Fitting requires first- or second-stage speed telemetry for synchronization.')
        self.output_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f'fit_case{target + 1}_', dir=self.output_root))
        self.fit_source_case = self.selected_case
        self.fit_telemetry = self.telemetry
        self.fit_options = dict(options, primary_group=self.telemetry_primary.get())
        data = self.telemetry
        snapshot = {'path': str(data.path), 'time': data.time.tolist(), 'time_note': data.time_note,
                    'groups': {group: {key: value.tolist() for key, value in channels.items()}
                               for group, channels in data.groups.items()}}
        configuration = dict(self.fit_options, base_request=request, telemetry=snapshot)
        (directory / 'fit_inputs.json').write_text(json.dumps(configuration, indent=2))
        source = self.cases[self.selected_case]
        if source.result is not None:
            with (directory / 'initial_solution.pickle').open('wb') as stream:
                pickle.dump(source.result, stream)
        (directory / 'case.json').write_text(json.dumps({'name': self.cases[target].name, 'color': self.cases[target].color}))
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        try:
            with (directory / 'solver.log').open('wb') as output:
                self.process = subprocess.Popen([sys.executable, str(PROJECT_DIR / 'DispersionFit.py'), '--worker', str(directory)],
                                                stdout=output, stderr=subprocess.STDOUT, cwd=PROJECT_DIR, env=env)
            self.log_stream = (directory / 'solver.log').open(errors='replace')
        except OSError:
            if self.process is not None:
                self.process.kill()
                self.process.wait()
                self.process = None
            raise
        self.run_dir = directory
        self.request = request
        self.submitted_inputs = self.capture_inputs()
        source.draft = deepcopy(self.submitted_inputs)
        self.job_kind = 'fit'
        self.running_case = target
        self.cases[target].state = 'Fitting (previous result retained)' if self.cases[target].result is not None else 'Fitting'
        self.cancelled = False
        self.started = time.monotonic()
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')
        self.append_log(f"Fitting {source.name} → {self.cases[target].name}. Files: {directory}\n")
        self.refresh_comparison()
        self.set_running(True)
        self.poll()

    def poll_fit_progress(self):
        text = f"{'Stopping fit' if self.cancelled else 'Fitting'} · {time.monotonic() - self.started:.0f} s"
        try:
            progress = json.loads((self.run_dir / 'fit_progress.json').read_text())
            self.progress.configure(value=progress['evaluation'])
            score = progress.get('best_score')
            text += f" · {progress['evaluation']}/{progress['max_evaluations']} evaluations"
            if score is not None:
                text += f" · best {score:.4g}"
        except (OSError, ValueError, KeyError):
            pass
        self.status.set(text)

    def finish_fit(self, case, code):
        try:
            path = self.run_dir / 'fit_result.pickle'
            if not path.exists():
                path = self.run_dir / 'fit_best.pickle'
            if not path.exists():
                raise RuntimeError('No successful synchronized candidate was saved. See solver output.')
            with path.open('rb') as stream:
                result = pickle.load(stream)
            if self.cancelled:
                result['termination'] = 'Cancelled; best completed comparison retained'
            elif code != 0:
                result['termination'] = 'Worker stopped early; best completed comparison retained (see solver output)'
            if path.name == 'fit_best.pickle':
                try:
                    result['history'] = json.loads((self.run_dir / 'fit_history.json').read_text())
                    result['evaluations'] = max(item['evaluation'] for item in result['history'])
                except (OSError, ValueError, KeyError):
                    pass
            report = fit_report(result)
            temporary_report = self.run_dir / 'fit_report.tmp'
            temporary_report.write_text(json.dumps(report, indent=2))
            temporary_report.replace(self.run_dir / 'fit_report.json')
            best = result['best']
            case.result = best['solution']
            case.request = deepcopy(best['request'])
            fitted = deepcopy(self.submitted_inputs)
            fitted['dispersion'] = {key: str(value) for key, value in best['request']['dispersion'].items()}
            fitted['payload_mass'] = str(best['request']['payload_mass_predefined'])
            case.computed_inputs = deepcopy(fitted)
            case.draft = fitted
            case.run_dir = self.run_dir
            case.fit_report = report
            case.state = f"Fit {'stopped' if self.cancelled or code != 0 else 'complete'} · score {best['score']:.4g}"
            source = self.cases[self.fit_source_case]
            if source is not case and source.result is None:
                source.result = result['baseline']['solution']
                source.request = deepcopy(result['baseline']['request'])
                source.computed_inputs = deepcopy(self.submitted_inputs)
                source.computed_inputs['payload_mass'] = str(source.request['payload_mass_predefined'])
                source.state = 'Baseline for fit'
            self.telemetry = self.fit_telemetry
            self.telemetry_primary.set(self.fit_options['primary_group'])
            self.telemetry_shift = best['synchronization']['shift']
            self.telemetry_shift_input.set(str(self.telemetry_shift))
            self.telemetry_visible.set(True)
            target = next(i for i, item in enumerate(self.cases) if item is case)
            self.case_visible[target].set(True)
            # Loading directly preserves the fitted draft when destination == source.
            self.load_case_form(target)
            self.refresh_comparison()
            self.notebook.select(self.tabs['Fit results'])
            self.status.set(f"{case.name}: {result['termination']} · score {result['baseline']['score']:.4g} → {best['score']:.4g}")
            self.append_log(f"\n{self.status.get()}\nReport: {self.run_dir / 'fit_report.json'}\n")
        except Exception as exc:
            case.state = ('Cancelled' if self.cancelled else 'Fit failed') + ('; previous result retained' if case.result is not None else '')
            self.status.set(f"{case.name}: {exc}")
            self.refresh_comparison()
            self.append_log(f"\n{exc}\n")
        self.save_state()

    def cancel(self):
        if self.process is not None and self.process.poll() is None:
            self.cancelled = True
            self.cancel_button.configure(state="disabled")
            self.process.terminate()
            process = self.process
            self.kill_id = self.root.after(1500, lambda: process.kill() if process.poll() is None else None)

    def close(self):
        self.save_state()
        self.closing = True
        if self.process is not None:
            self.cancel()
        else:
            self.destroy_window()

    def destroy_window(self):
        # Tk does not remove pending after scripts when their Python callbacks
        # disappear. Cancel our timers and queued Matplotlib draws explicitly.
        for callback in (self.save_id, self.poll_id, self.kill_id, self.layout_id):
            if callback is not None:
                self.root.after_cancel(callback)
        for canvas in self.canvases.values():
            if canvas._idle_draw_id is not None:
                canvas.get_tk_widget().after_cancel(canvas._idle_draw_id)
                canvas._idle_draw_id = None
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    import tkinter as tk
    root = tk.Tk()
    OptimizationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
