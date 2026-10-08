"""Read telemetry CSVs and overlay recognized measurements in GUI plot units."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from Atmosphere_Type import AtmosphereType
from LV_Type_scaled import VLType


# Preferred columns first; compatible alternatives fill missing samples only.
CHANNELS = {
    "altitude": [("altitude_km", 1), ("altitude_m", .001), ("alt_km_altitude_roi", 1), ("alt_km_primary", 1)],
    "speed": [("speed_mps", 1), ("velocity_mps", 1), ("speed_kmh", 1/3.6), ("velocity_kmh", 1/3.6), ("vel_kmh_speed", 1/3.6), ("vel_kmh_primary", 1/3.6)],
    "acceleration": [("specific_acceleration_mps2", 1), ("acceleration_mps2", 1), ("acceleration_g", 9.80665), ("accel_g", 9.80665), ("accel_g_acceleration", 9.80665), ("accel_g_primary", 9.80665)],
    "mass": [("mass_kg", .001), ("mass_t", 1)],
    "thrust_factor": [("thrust_factor", 1), ("throttle", 1), ("throttle_percent", .01)],
    "alpha": [("angle_of_attack_deg", 1), ("alpha_deg", 1)],
    "pressure": [("dynamic_pressure_kpa", 1), ("dynamic_pressure_pa", .001)],
    "heat_flux": [("heat_flux_kw_m2", 1), ("heat_flux_w_m2", .001)],
    "isp": [("isp_s", 1), ("specific_impulse_s", 1)],
    "vx": [("ecef_vx_mps", 1), ("vx_mps", 1)],
    "vy": [("ecef_vy_mps", 1), ("vy_mps", 1)],
    "vz": [("ecef_vz_mps", 1), ("vz_mps", 1)],
    "flight_path": [("flight_path_angle_deg", 1)],
    "thrust": [("thrust_kn", 1), ("thrust_n", .001)],
    "inclination": [("inclination_deg", 1)],
    "apogee": [("apogee_altitude_km", 1), ("apogee_km", 1)],
    "perigee": [("perigee_altitude_km", 1), ("perigee_km", 1)],
    "thrust_angle": [("thrust_velocity_angle_deg", 1)],
    "thrust_x": [("thrust_x_kn", 1), ("thrust_x_n", .001)],
    "thrust_y": [("thrust_y_kn", 1), ("thrust_y_n", .001)],
    "thrust_z": [("thrust_z_kn", 1), ("thrust_z_n", .001)],
    "radius": [("radius_km", 1), ("radius_m", .001)],
    "radial": [("radial_velocity_mps", 1)],
    "transverse": [("transverse_speed_mps", 1)],
    "eccentricity": [("eccentricity", 1)],
    "energy": [("specific_energy_mj_kg", 1), ("specific_energy_j_kg", 1e-6)],
    "downrange": [("downrange_km", 1), ("downrange_m", .001)],
    "local_z": [("local_z_km", 1), ("local_z_m", .001)],
}
PREFIXES = {"primary": ("",), "stage1": ("stage1_", "s1_", "first_stage_"),
            "stage2": ("stage2_", "s2_", "second_stage_"), "booster": ("booster_",)}
PRIMARY_GROUPS = ('Phase columns', 'Stage 1 / booster', 'Stage 2', 'Booster')
# The numbered broadcast columns identify the vehicle, not successive time
# windows: upper-stage and booster readings coexist after stage separation.
NUMBERED_CHANNELS = {
    phase: {
        'speed': [(f'velocity{number}_mps', 1), (f'speed{number}_mps', 1),
                  (f'velocity{number}_kmh', 1/3.6), (f'speed{number}_kmh', 1/3.6)],
        'altitude': [(f'altitude{number}_km', 1), (f'altitude{number}_m', .001)],
        'acceleration': [(f'specific_acceleration{number}_mps2', 1),
                         (f'acceleration{number}_mps2', 1), (f'acceleration{number}_g', 9.80665)],
        'pressure': [(f'dynamic_pressure{number}_kpa', 1), (f'dynamic_pressure{number}_pa', .001)],
    }
    for phase, number in (('stage1', 1), ('stage2', 2), ('booster', 3))
}
# Axis, channel, title, y unit. Trajectories use a separate x channel.
PLOTS = {
    "First stage": [(0, "local_z", "Local trajectory", "Local Z [km]"), (1, "altitude", "Altitude", "Altitude [km]"), (2, "speed", "ECEF speed", "Speed [m/s]"), (3, "mass", "Vehicle mass", "Mass [t]"), (4, "thrust_factor", "Engine & steering", "Thrust factor [−]"), (5, "pressure", "Dynamic pressure", "Pressure [kPa]")],
    "S1 details": [(0, "isp", "Specific impulse", "Isp [s]"), (1, "acceleration", "Specific acceleration", "Acceleration [m/s²]"), (2, "alpha", "Angle of attack", "Angle [deg]"), (3, "vx", "ECEF velocity X", "Velocity X [m/s]"), (4, "vz", "ECEF velocity Z", "Velocity Z [m/s]"), (5, "flight_path", "Flight angle to local X axis", "Angle [deg]")],
    "Second stage": [(0, "altitude", "Altitude", "Altitude [km]"), (1, "speed", "ECEF speed", "Speed [m/s]"), (2, "mass", "Vehicle mass", "Mass [t]"), (3, "thrust", "Engine thrust", "Thrust [kN]"), (4, "inclination", "Orbital inclination", "Inclination [deg]"), (5, "apogee", "Osculating orbit", "Altitude [km]"), (5, "perigee", "Osculating orbit", "Altitude [km]")],
    "S2 propulsion": [(0, "isp", "Specific impulse", "Isp [s]"), (1, "acceleration", "Specific acceleration", "Acceleration [m/s²]"), (2, "thrust_angle", "Thrust / ECI velocity angle", "Angle [deg]"), (3, "thrust_x", "ECI thrust X", "Thrust [kN]"), (4, "thrust_y", "ECI thrust Y", "Thrust [kN]"), (5, "thrust_z", "ECI thrust Z", "Thrust [kN]")],
    "S2 orbit": [(0, "radius", "Geocentric radius", "Radius [km]"), (1, "radial", "Radial velocity", "Velocity [m/s]"), (2, "transverse", "Transverse speed", "Speed [m/s]"), (3, "flight_path", "Flight angle to local horizon", "Angle [deg]"), (4, "eccentricity", "Orbital eccentricity", "Eccentricity [−]"), (5, "energy", "Specific orbital energy", "Energy [MJ/kg]")],
    "Booster": [(0, "local_z", "Return trajectory", "Local Z [km]"), (1, "altitude", "Altitude", "Altitude [km]"), (2, "speed", "ECEF speed", "Speed [m/s]"), (3, "mass", "Booster mass", "Mass [t]"), (4, "acceleration", "Specific acceleration", "Acceleration [m/s²]"), (5, "pressure", "Dynamic pressure", "Pressure [kPa]")],
}


# Keep supplied measurements distinct from estimates derived from altitude/speed.
for _phase in ("First stage", "Booster"):
    PLOTS[_phase].append((5, "pressure_derived", "Dynamic pressure", "Pressure [kPa]"))
for _phase, _axis in (("S2 propulsion", 1), ("Booster", 4)):
    _title = "Speed acceleration (ECEF, includes gravity)" if _phase == "S2 propulsion" else "Acceleration: specific / dv/dt"
    PLOTS[_phase].append((_axis, "acceleration_derived", _title, "Acceleration [m/s²]"))
PLOTS["Booster"].extend([
    (6, "heat_flux", "Empirical heat flux", "Heat flux [kW/m²]"),
    (6, "heat_flux_derived", "Empirical heat flux", "Heat flux [kW/m²]"),
])


def make_plot_axes(figure, phase):
    """Keep the six existing booster plots and add a wide heating plot below."""
    if phase == "Booster":
        grid = figure.add_gridspec(2, 4)
        return ([figure.add_subplot(grid[0, col]) for col in range(4)]
                + [figure.add_subplot(grid[1, col]) for col in range(2)]
                + [figure.add_subplot(grid[1, 2:])])
    return figure.subplots(2, 3).ravel()


def aerodynamic_loads(altitude_m, speed_mps, density_factor=1.0,
                      heat_coefficient=VLType.Booster_k_empirical):
    """Use the project's log-density atmosphere and empirical heating relation.

    Outputs are kPa and kW/m². Speed is atmosphere-relative (Earth-fixed in
    the no-wind model). Invalid samples remain gaps rather than interpolating.
    """
    altitude, speed = np.broadcast_arrays(np.asarray(altitude_m, dtype=float),
                                         np.asarray(speed_mps, dtype=float))
    pressure, heating = np.full(altitude.shape, np.nan), np.full(altitude.shape, np.nan)
    valid = np.isfinite(altitude) & np.isfinite(speed) & (speed >= 0)
    if valid.any():
        rho = np.asarray(AtmosphereType.rho_fun(altitude[valid]), dtype=float).reshape(-1) * density_factor
        with np.errstate(over="ignore", invalid="ignore"):
            pressure[valid] = .5 * rho * speed[valid]**2 / 1000
            heating[valid] = heat_coefficient * np.sqrt(rho) * speed[valid]**3 / 1000
    return {"pressure": np.where(np.isfinite(pressure), pressure, np.nan),
            "heat_flux": np.where(np.isfinite(heating), heating, np.nan)}


def numerical_acceleration(time_s, speed_mps):
    """Signed speed derivative in its input frame, using backward differences in m/s².

    Each value belongs to the later sample: (v[i] - v[i-1]) / (t[i] - t[i-1]).
    The first sample and invalid, duplicate-time, or reversed intervals remain
    gaps. Missing samples are never bridged and no smoothing is applied.
    """
    time = np.asarray(time_s, dtype=float).ravel()
    speed = np.asarray(speed_mps, dtype=float).ravel()
    if time.size != speed.size:
        raise ValueError("Telemetry time and speed must have the same length.")
    acceleration = np.full(time.shape, np.nan)
    valid = np.isfinite(time) & np.isfinite(speed) & (speed >= 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dt, dv = np.diff(time), np.diff(speed)
        intervals = valid[:-1] & valid[1:] & (dt > 0) & np.isfinite(dt)
        np.divide(dv, dt, out=acceleration[1:], where=intervals)
    acceleration[~np.isfinite(acceleration)] = np.nan
    return acceleration


def mission_seconds(value):
    text = str(value).strip().upper().replace("−", "-")
    match = re.fullmatch(r"(?:T\s*)?([+-]?)(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    if not match:
        return np.nan
    sign, hours, minutes, seconds = match.groups()
    if int(minutes) >= 60 or float(seconds) >= 60:
        return np.nan
    return (-1 if sign == "-" else 1) * (3600 * int(hours) + 60 * int(minutes) + float(seconds))


@dataclass
class TelemetryData:
    path: Path
    time: np.ndarray
    groups: dict
    time_note: str

    def channels(self, group, primary_group):
        channels = dict(self.groups[group])
        use_primary = (primary_group == "Stage 1 / booster" and group in ("stage1", "booster")
                       or primary_group == "Stage 2" and group == "stage2"
                       or primary_group == "Booster" and group == "booster")
        if use_primary:
            for key, value in self.groups["primary"].items():
                channels[key] = np.where(np.isfinite(channels[key]), channels[key], value) if key in channels else value
        # All velocity telemetry is ECEF. Prefer complete vectors when present;
        # scalar speed remains usable where one or more components are missing.
        if all(key in channels for key in ("vx", "vy", "vz")):
            vectors = np.array([channels[key] for key in ("vx", "vy", "vz")])
            valid_vectors = np.isfinite(vectors).all(axis=0)
            channels["speed"] = np.where(valid_vectors, np.linalg.norm(vectors, axis=0),
                                         channels.get("speed", np.full(len(self.time), np.nan)))
        if group != "stage1" and "speed" in channels:
            acceleration = numerical_acceleration(self.time, channels["speed"])
            if np.isfinite(acceleration).any():
                channels["acceleration_derived"] = acceleration
        if group in ("stage1", "booster") and "altitude" in channels and "speed" in channels:
            loads = aerodynamic_loads(channels["altitude"] * 1000, channels["speed"])
            for key, values in loads.items():
                if np.isfinite(values).any():
                    channels[key + "_derived"] = values
        return channels


def load_telemetry_csv(path):
    path = Path(path)
    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Telemetry CSV is empty or has duplicate column names.")

    def numeric(name):
        if name not in frame:
            return np.full(len(frame), np.nan)
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, np.nan)

    time = np.full(len(frame), np.nan)
    for name in ("time_s", "mission_time_s", "time_sec", "time_seconds"):
        time = np.where(np.isfinite(time), time, numeric(name))
    for name in ("time_str_primary", "mission_time", "time_str"):
        if name in frame:
            time = np.where(np.isfinite(time), time, frame[name].map(mission_seconds).to_numpy())
    video = numeric("t_video_sec")
    note = "Mission time"
    paired = np.isfinite(time) & np.isfinite(video)
    if paired.any():
        fallback = video - np.median(video[paired] - time[paired])
        time = np.where(np.isfinite(time), time, fallback)
    elif not np.isfinite(time).any() and np.isfinite(video).any():
        time = video - np.nanmin(video)
        note = "Video-relative time; adjust shift to align with launch"
    if not np.isfinite(time).any():
        raise ValueError("No usable time column. Use time_s, time_str_primary (HH:MM:SS), or t_video_sec.")
    valid = np.isfinite(time)
    order = np.flatnonzero(valid)[np.argsort(time[valid], kind="stable")]
    groups = {}
    for group, prefixes in PREFIXES.items():
        channels = {}
        for key, aliases in CHANNELS.items():
            values = np.full(len(frame), np.nan)
            names = (NUMBERED_CHANNELS.get(group, {}).get(key, [])
                     + [(prefix + name, scale) for prefix in prefixes for name, scale in aliases])
            for name, scale in names:
                values = np.where(np.isfinite(values), values, numeric(name) * scale)
            if np.isfinite(values[order]).any():
                channels[key] = values[order]
        groups[group] = channels
    if not any(groups.values()):
        raise ValueError("No recognized telemetry measurements with valid times. Use unit-labelled columns such as altitude_km or speed_kmh.")
    return TelemetryData(path, time[order], groups, note)


def interpolate_speed_time(time, speed, target_speed):
    """First rising crossing in chronological order, interpolated in speed.

    Invalid readings are skipped. Equal speeds use the first matching sample;
    duplicate timestamps cannot form an interpolation interval. No extrapolation.
    """
    time, speed = np.asarray(time, dtype=float).ravel(), np.asarray(speed, dtype=float).ravel()
    if time.size != speed.size or not np.isfinite(target_speed) or target_speed < 0:
        raise ValueError("Invalid time/speed samples for telemetry synchronization.")
    valid = np.isfinite(time) & np.isfinite(speed) & (speed >= 0)
    order = np.argsort(time[valid], kind="stable")
    time, speed = time[valid][order], speed[valid][order]
    for index in range(len(time)):
        if speed[index] == target_speed:
            return float(time[index])
        if (index + 1 < len(time) and time[index + 1] > time[index]
                and speed[index] < target_speed < speed[index + 1]):
            fraction = (target_speed - speed[index]) / (speed[index + 1] - speed[index])
            return float(time[index] + fraction * (time[index + 1] - time[index]))
    raise ValueError(f"No ascent telemetry samples bracket the initial speed ({target_speed:.3f} m/s). "
                     "Load telemetry covering liftoff and early ascent.")


def phase_group(phase):
    return "booster" if phase == "Booster" else "stage2" if phase.startswith("S2") or phase == "Second stage" else "stage1"


def overlay_telemetry(figure, phase, telemetry, primary_group, color, shift=0, windows=None):
    """Overlay measurements and available load estimates, preserving sample gaps."""
    channels = telemetry.channels(phase_group(phase), primary_group)
    time = telemetry.time + shift
    in_window = np.ones(time.size, dtype=bool)
    if windows:
        in_window = np.zeros(time.size, dtype=bool)
        for start, end in windows:
            in_window |= (time >= start) & (time <= end)
    plots = []
    for index, key, title, ylabel in PLOTS[phase]:
        if key not in channels:
            continue
        trajectory = key == "local_z"
        if trajectory and "downrange" not in channels:
            continue
        x = channels["downrange"] if trajectory else time
        channel_window = in_window
        if key == "acceleration_derived" and windows:
            # Both endpoints must belong to the same flight-phase window.
            channel_window = np.zeros(time.size, dtype=bool)
            for start, end in windows:
                channel_window[1:] |= (time[:-1] >= start) & (time[1:] <= end)
        y = np.where(channel_window, channels[key], np.nan)
        if (np.isfinite(x) & np.isfinite(y)).any():
            plots.append((index, key, title, ylabel, x, y, trajectory))
    if not plots:
        return False
    standalone = len(figure.axes) == 0
    axes = make_plot_axes(figure, phase) if standalone else figure.axes[:7 if phase == "Booster" else 6]
    used = set()
    for index, key, title, ylabel, x, y, trajectory in plots:
        ax = axes[index]
        label = "Telemetry" + (f" · {key}" if key in ("apogee", "perigee") else "")
        if key == "acceleration_derived":
            label += " · dv/dt (ECEF)"
            ax.set_title(title)
        elif key.endswith("_derived"):
            label += " · derived"
        elif key in ("pressure", "heat_flux", "acceleration"):
            label += " · CSV"
        ax.plot(x, y, color=color, linestyle=":" if key in ("perigee", "pressure", "heat_flux", "acceleration") else "--", marker=".",
                markersize=3, linewidth=1.2, label=label)
        if standalone:
            ax.set(title=title, xlabel="Downrange [km]" if trajectory else "Time [s]", ylabel=ylabel)
            ax.grid(True, alpha=.25)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7)
        used.add(index)
    if standalone:
        for index, ax in enumerate(axes):
            if index not in used:
                ax.set_axis_off()
                ax.text(.5, .5, "No matching telemetry channel", transform=ax.transAxes, ha="center", fontsize=9, color="gray")
    return True
