#!/usr/bin/env python3
"""
Compare a simulated Falcon 9 ascent profile against extracted telemetry.

The script:
1) Loads the OCR'd telemetry CSV from ExtractTelemetry.py.
2) Runs a nominal LV_Optimization simulation (Falcon 9, ASDS by default).
3) Overlays altitude and velocity time histories and reports simple RMSE metrics.
"""
import os
os.system('clear')
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import LV_Optimization_Type as LVopt_Type
from LV_Type_scaled import DispesrionFactorsType


OUTPUT_DIR = Path("falcon9_telemetry")
TELEMETRY_FILE = OUTPUT_DIR / "falcon9_telemetry_starlink.csv"

# Earth radius (matches Main.py)
R0 = 6_378_137.0  # [m]

# Manual time shift applied to simulated profiles (seconds).
# Positive values delay the sim timeline; negative values move it earlier.
SIM_TIME_SHIFT = 0.0


def _parse_time_str_to_seconds(text: str) -> float:
    """Parse a HH:MM:SS string (optionally prefixed with T+/T-) to seconds."""
    if not isinstance(text, str) or not text:
        return math.nan
    cleaned = text.strip().lower().replace("t+", "").replace("t-", "")
    parts = cleaned.split(":")
    if len(parts) != 3:
        return math.nan
    try:
        hh, mm, ss = [int(p) for p in parts]
        return 3600 * hh + 60 * mm + ss
    except ValueError:
        return math.nan


def load_telemetry(csv_path: Path) -> pd.DataFrame:
    """Load telemetry CSV and return a cleaned DataFrame with time/alts/vels."""
    df = pd.read_csv(csv_path)

    # Primary mission time column
    df["time_s"] = df["time_str_primary"].apply(_parse_time_str_to_seconds)
    video_time = df["t_video_sec"] - df["t_video_sec"].min()
    df["time_s"] = df["time_s"].fillna(video_time)

    # Convert units and gently interpolate through OCR gaps
    df["altitude_km"] = pd.to_numeric(df["altitude_km"], errors="coerce")
    df["velocity_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
    df["accel_g"] = pd.to_numeric(df["acceleration_g"], errors="coerce")

    for col in ("altitude_km", "velocity_kmh", "accel_g"):
        df[col] = df[col].interpolate(limit_direction="both")

    # Drop rows without time
    df = df.dropna(subset=["time_s"])
    df = df.sort_values("time_s").reset_index(drop=True)
    return df


def run_nominal_simulation():
    """Run LVopt in a Falcon 9 / ASDS / LEO configuration."""
    disp = DispesrionFactorsType()
    disp.FirstStage_EmptyMass = 2500
    disp.FirstStage_PropMass = -7500
    disp.FirstStageThrust = 0.96
    disp.FirstStageIsp = 1.0
    disp.SecondStageIsp = 1.0
    disp.SecondStageThrust = 0.90
    disp.SecondStage_EmptyMass = 500
    disp.SecondStage_PropMass = -500
    lvopt = LVopt_Type.LV_Optimization(disp, LV_Configuration=1)
    

    target_orbit = {
        "Name": "LEO",
        "apogee": R0 + 248_000.0,
        "perigee": R0 + 264_000.0,
        "i": np.deg2rad(70.0),
    }

    sol = lvopt.SolveOptimiztion(
        RecoveryStrategy="ASDS",
        payload_mass_predefined=15500,
        Target_Orbit=target_orbit,
        Plot_interm=False,
        print_ipopt=True,
        init_guess=0.5,
    )
    if not sol.get("success", False):
        raise RuntimeError(f"Simulation failed: {sol.get('return_status')}")

    first_stage_propellant_at_separation = float(sol["x4_0"][4]) - lvopt.rocket_return.EmptyFirstStageMass
    first_stage_propellant_at_landing = float(sol["x4_landing"][4, -1]) - lvopt.rocket_return.EmptyFirstStageMass
    # The final upper-stage state includes payload; the fairing is already jettisoned.
    second_stage_propellant_remaining = (
        float(sol["x3"][6, -1]) - lvopt.eci.EmptySecondStageMass - float(sol["payload_mass"])
    )
    print(f"First-stage propellant remaining at separation: {first_stage_propellant_at_separation:,.1f} kg")
    print(f"First-stage propellant remaining after landing: {first_stage_propellant_at_landing:,.1f} kg")
    print(f"Second-stage propellant remaining at final simulated cutoff "
          f"(before the final orbit-adjustment burn): {second_stage_propellant_remaining:,.1f} kg")
    second_stage_propellant_after_adjustment = (
        second_stage_propellant_remaining - sol["propellant_mass_for_final_dv"]
    )
    print(f"Second-stage propellant remaining after the orbit-adjustment burn: "
          f"{second_stage_propellant_after_adjustment:,.1f} kg")
    return lvopt, sol


def _local_to_ecef_rot(lat_deg: float, lon_deg: float, az_rad: float) -> np.ndarray:
    """Rotation matrix from local (x along az, y completing RH, z up) to ECEF."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    # Local Up (Z)
    z = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=float)
    # Local North
    n = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=float)
    # Local East
    e = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=float)
    # X-axis along azimuth
    x = np.cos(az_rad) * n + np.sin(az_rad) * e
    # Y-axis completes RH
    y = np.array([z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2], z[0]*x[1] - z[1]*x[0]], dtype=float)
    return np.column_stack((x, y, z))


def _local_vel_to_ecef(vx: np.ndarray, vz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    v_local = np.vstack((vx, np.zeros_like(vx), vz))
    return rot @ v_local


def _eci_vel_to_ecef(v_eci: np.ndarray, r_eci: np.ndarray, omega_earth: float) -> np.ndarray:
    """Convert inertial velocity to Earth-fixed velocity via v_ecef = v_eci - ω×r."""
    omega_vec = np.array([0.0, 0.0, omega_earth], dtype=float)[:, None]
    cross = np.cross(omega_vec.T, r_eci.T).T
    return v_eci - cross


def _simulation_heat_flux_kw_m2(states: np.ndarray, model) -> np.ndarray:
    """Evaluate the model's heat flux for SI state rows and convert W/m² to kW/m²."""
    return np.array([
        float(model.heat_flux_fun(model.scale_x(state))) for state in states
    ]) / 1000.0


def simulation_profile_to_df(sol: dict, lvopt) -> pd.DataFrame:
    """Extract first-stage motion, dynamic pressure, and model heat flux."""
    t = np.asarray(sol["t1_vec"], dtype=float)
    alt_km = np.asarray(sol["alt1_sol"], dtype=float) / 1000.0
    rot = _local_to_ecef_rot(lvopt.eci.LaunchLatitude, lvopt.eci.LaunchLongitude, sol["LaunchAz"])
    v_ecef = _local_vel_to_ecef(sol["x1"][2, :], sol["x1"][3, :], rot)
    vel_kmh = np.linalg.norm(v_ecef, axis=0) * 3.6
    acc_raw = _pad_acc_like_time(np.asarray(sol.get("acc1_sol", []), dtype=float), t.size, vel_kmh)
    acc_g = acc_raw / 9.80665
    return pd.DataFrame(
        {
            "time_s": t,
            "altitude_km": alt_km,
            "velocity_kmh": vel_kmh,
            "accel_g": acc_g,
            "dynamic_pressure_kpa": np.asarray(sol["Qdyn1_sol"], dtype=float).reshape(-1) / 1000.0,
            "heat_flux_kw_m2": _simulation_heat_flux_kw_m2(np.asarray(sol["x1"]).T, lvopt.booster),
        }
    )


def align_sim_time_by_velocity(sim_df: pd.DataFrame, tel_df: pd.DataFrame):
    """
    Shift sim time so the FIRST simulated point lines up with the telemetry timestamp
    interpolated at that point's velocity, using the first bracketing pair before
    the first telemetry reading above 200 km/h. Out-of-range speeds are clamped
    to the closest available speed; repeated speeds use their first occurrence.
    Returns (aligned_df, applied_shift) where applied_shift is subtracted from sim time.
    """
    sim_vel = sim_df["velocity_kmh"].to_numpy()
    sim_time = sim_df["time_s"].to_numpy()
    if sim_vel.size == 0 or not np.isfinite(sim_vel[0]) or not np.isfinite(sim_time[0]):
        return sim_df, 0.0
    sim_first_vel = sim_vel[0]

    tel_mask = np.isfinite(tel_df["velocity_kmh"])
    if not tel_mask.any():
        return sim_df, 0.0
    tel_vel = tel_df.loc[tel_mask, "velocity_kmh"].to_numpy()
    tel_time = tel_df.loc[tel_mask, "time_s"].to_numpy()

    above_200 = np.flatnonzero(tel_vel > 200.0)
    if above_200.size:
        cutoff = int(above_200[0])
        tel_vel = tel_vel[:cutoff]
        tel_time = tel_time[:cutoff]
    if tel_vel.size == 0:
        return sim_df, 0.0

    valid = np.isfinite(tel_time)
    tel_vel, tel_time = tel_vel[valid], tel_time[valid]
    if tel_vel.size == 0:
        return sim_df, 0.0

    # Find the first crossing in time order, even if OCR speeds are not monotonic.
    crossings = np.flatnonzero(
        ((tel_vel[:-1] <= sim_first_vel) & (sim_first_vel <= tel_vel[1:]))
        | ((tel_vel[1:] <= sim_first_vel) & (sim_first_vel <= tel_vel[:-1]))
    )
    if crossings.size:
        idx = int(crossings[0])
        delta_vel = tel_vel[idx + 1] - tel_vel[idx]
        fraction = (sim_first_vel - tel_vel[idx]) / delta_vel if delta_vel else 0.0
        target_time = tel_time[idx] + fraction * (tel_time[idx + 1] - tel_time[idx])
    else:
        # No bracket: clamp to the available speed range instead of extrapolating.
        idx = int(np.argmin(tel_vel) if sim_first_vel < tel_vel.min() else np.argmax(tel_vel))
        target_time = tel_time[idx]
    sim_first_time = sim_time[0]
    shift = sim_first_time - target_time

    sim_aligned = sim_df.copy()
    applied_shift = shift - SIM_TIME_SHIFT
    sim_aligned["time_s"] = sim_aligned["time_s"] - applied_shift
    return sim_aligned, applied_shift


def simulation_stage2_profile(sol: dict, lvopt, time_shift: float = 0.0) -> pd.DataFrame:
    """Build combined second-stage (before+after fairing) profile in ECEF (shifted by time_shift)."""
    t2 = np.asarray(sol["t2_vec"], dtype=float)
    t3 = np.asarray(sol["t3_vec"], dtype=float)
    t = np.concatenate([t2, t3[1:]])  # avoid duplicate knot
    alt_km = np.concatenate([np.asarray(sol["alt2_sol"])/1000.0, np.asarray(sol["alt3_sol"][1:])/1000.0])
    v2_ecef = _eci_vel_to_ecef(sol["x2"][3:6, :], sol["x2"][0:3, :], lvopt.eci.omega_earth)
    v3_ecef = _eci_vel_to_ecef(sol["x3"][3:6, :], sol["x3"][0:3, :], lvopt.eci.omega_earth)
    vel2 = np.linalg.norm(v2_ecef, axis=0)
    vel3 = np.linalg.norm(v3_ecef, axis=0)
    vel_kmh = np.concatenate([vel2, vel3[1:]]) * 3.6
    acc2 = _pad_acc_like_time(np.asarray(sol.get("acc2_sol", [])), t2.size, vel2)
    acc3 = _pad_acc_like_time(np.asarray(sol.get("acc3_sol", [])), t3.size, vel3)
    acc_raw = np.concatenate([acc2, acc3[1:]])
    acc_g = acc_raw / 9.80665
    return pd.DataFrame(
        {
            "time_s": t - time_shift,
            "altitude_km": alt_km,
            "velocity_kmh": vel_kmh,
            "accel_g": acc_g,
        }
    )


def simulation_booster_profile(sol: dict, lvopt, time_shift: float = 0.0) -> pd.DataFrame:
    """Booster return (if present), shifted by time_shift and expressed in ECEF speed."""
    if "t4_vec" not in sol or "x4" not in sol:
        return pd.DataFrame()
    t = np.asarray(sol["t4_vec"], dtype=float).reshape(-1)
    alt_km = np.asarray(sol["alt4_sol"], dtype=float).reshape(-1) / 1000.0
    rot = _local_to_ecef_rot(lvopt.eci.LaunchLatitude, lvopt.eci.LaunchLongitude, sol["LaunchAz"])
    v_ecef = _local_vel_to_ecef(sol["x4"][:, 2], sol["x4"][:, 3], rot).T
    vel_kmh = np.linalg.norm(v_ecef, axis=1).reshape(-1) * 3.6
    acc_raw = _pad_acc_like_time(np.asarray(sol.get("acc4_sol", [])), t.size, vel_kmh).reshape(-1)
    acc_g = acc_raw / 9.80665
    return pd.DataFrame(
        {
            "time_s": t - time_shift,
            "altitude_km": alt_km,
            "velocity_kmh": vel_kmh,
            "accel_g": acc_g,
            "dynamic_pressure_kpa": np.asarray(sol["Qdyn4_sol"], dtype=float).reshape(-1) / 1000.0,
            "heat_flux_kw_m2": _simulation_heat_flux_kw_m2(np.asarray(sol["x4"]), lvopt.rocket_return),
        }
    )


def _pad_acc_like_time(acc_raw: np.ndarray, time_len: int, vel_ref: np.ndarray) -> np.ndarray:
    """Align acceleration array to time length."""
    acc_raw = np.asarray(acc_raw, dtype=float)
    if acc_raw.size == time_len - 1:
        acc_raw = np.concatenate([acc_raw, acc_raw[-1:]])
    elif acc_raw.size != time_len:
        acc_raw = np.full_like(vel_ref, np.nan)
    return acc_raw


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return math.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def compare_and_plot(sim_df: pd.DataFrame, tel_df: pd.DataFrame, stage_label: str = "First Stage"):
    """Overlay profiles and print RMSE stats on a common time grid."""
    t_end = min(sim_df["time_s"].max(), tel_df["time_s"].max())
    common_time = np.linspace(0.0, t_end, 600)

    sim_alt = np.interp(common_time, sim_df["time_s"], sim_df["altitude_km"])
    tel_alt = np.interp(common_time, tel_df["time_s"], tel_df["altitude_km"])
    sim_vel = np.interp(common_time, sim_df["time_s"], sim_df["velocity_kmh"])
    tel_vel = np.interp(common_time, tel_df["time_s"], tel_df["velocity_kmh"])
    sim_acc = np.interp(common_time, sim_df["time_s"], sim_df["accel_g"])
    tel_acc = np.interp(common_time, tel_df["time_s"], tel_df["accel_g"])

    alt_rmse = compute_rmse(sim_alt, tel_alt)
    vel_rmse = compute_rmse(sim_vel, tel_vel)
    acc_rmse = compute_rmse(sim_acc, tel_acc)
    print(f"Altitude RMSE: {alt_rmse:.2f} km")
    print(f"Velocity RMSE: {vel_rmse:.2f} km/h")
    print(f"Acceleration RMSE: {acc_rmse:.3f} g")

    fig, (ax_alt, ax_vel, ax_acc) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    ax_alt.plot(sim_df["time_s"], sim_df["altitude_km"], label="Sim (LVopt)", lw=2)
    ax_alt.plot(tel_df["time_s"], tel_df["altitude_km"],'-s', label="Telemetry", lw=2, alpha=0.7)
    ax_alt.set_ylabel("Altitude [km]")
    ax_alt.set_title(f"Falcon 9 {stage_label} Altitude")
    ax_alt.grid(True)
    ax_alt.legend()

    ax_vel.plot(sim_df["time_s"], sim_df["velocity_kmh"], label="Sim (LVopt)", lw=2)
    ax_vel.plot(tel_df["time_s"], tel_df["velocity_kmh"],'s-', label="Telemetry", lw=2, alpha=0.7)
    ax_vel.set_ylabel("Velocity [km/h]")
    ax_vel.set_title(f"Falcon 9 {stage_label} Velocity")
    ax_vel.grid(True)
    ax_vel.legend()

    ax_acc.plot(sim_df["time_s"], sim_df["accel_g"], label="Sim (LVopt)", lw=2)
    ax_acc.plot(tel_df["time_s"], tel_df["accel_g"],'s-', label="Telemetry", lw=2, alpha=0.7)
    ax_acc.set_ylabel("Acceleration [g]")
    ax_acc.set_xlabel("Time since liftoff [s]")
    ax_acc.set_title(f"Falcon 9 {stage_label} Acceleration")
    ax_acc.grid(True)
    ax_acc.legend()

    fig.tight_layout()


def plot_stage1_and_booster(sim_stage1: pd.DataFrame, tel_stage1: pd.DataFrame,
                            sim_booster: pd.DataFrame, *, rho_fun, heat_flux_coefficient):
    """Plot motion, dynamic pressure, and heat flux from simulation and telemetry."""
    altitude_m = tel_stage1["altitude_km"].to_numpy(dtype=float) * 1000.0
    velocity_mps = tel_stage1["velocity_kmh"].to_numpy(dtype=float) / 3.6
    telemetry_q_kpa = np.full(altitude_m.shape, np.nan)
    telemetry_heat_kw_m2 = np.full(altitude_m.shape, np.nan)
    valid = np.isfinite(altitude_m) & np.isfinite(velocity_mps)
    if valid.any():
        rho = np.asarray(rho_fun(altitude_m[valid]), dtype=float).reshape(-1)
        telemetry_q_kpa[valid] = 0.5 * rho * velocity_mps[valid] ** 2 / 1000.0
        telemetry_heat_kw_m2[valid] = heat_flux_coefficient * np.sqrt(rho) * velocity_mps[valid] ** 3 / 1000.0

    fig, (ax_alt, ax_vel, ax_acc, ax_q, ax_heat) = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

    ax_alt.plot(sim_stage1["time_s"], sim_stage1["altitude_km"], label="Sim Stage 1", lw=2)
    ax_alt.plot(tel_stage1["time_s"], tel_stage1["altitude_km"], label="Telemetry Stage 1", lw=2, alpha=0.7)
    ax_alt.plot(sim_booster["time_s"], sim_booster["altitude_km"], label="Sim Booster Return", lw=2, ls="--")
    ax_alt.set_ylabel("Altitude [km]")
    ax_alt.set_title("Stage 1 & Booster Return Altitude")
    ax_alt.grid(True)
    ax_alt.legend()

    ax_vel.plot(sim_stage1["time_s"], sim_stage1["velocity_kmh"], label="Sim Stage 1", lw=2)
    ax_vel.plot(tel_stage1["time_s"], tel_stage1["velocity_kmh"], label="Telemetry Stage 1", lw=2, alpha=0.7)
    ax_vel.plot(sim_booster["time_s"], sim_booster["velocity_kmh"], label="Sim Booster Return", lw=2, ls="--")
    ax_vel.set_ylabel("Velocity [km/h]")
    ax_vel.set_title("Stage 1 & Booster Return Velocity")
    ax_vel.grid(True)
    ax_vel.legend()

    ax_acc.plot(sim_stage1["time_s"], sim_stage1["accel_g"], label="Sim Stage 1", lw=2)
    ax_acc.plot(tel_stage1["time_s"], tel_stage1["accel_g"], label="Telemetry Stage 1", lw=2, alpha=0.7)
    ax_acc.plot(sim_booster["time_s"], sim_booster["accel_g"], label="Sim Booster Return", lw=2, ls="--")
    ax_acc.set_ylabel("Acceleration [g]")
    ax_acc.set_title("Stage 1 & Booster Return Acceleration")
    ax_acc.grid(True)
    ax_acc.legend()

    ax_q.plot(sim_stage1["time_s"], sim_stage1["dynamic_pressure_kpa"], label="Sim Stage 1", lw=2)
    ax_q.plot(tel_stage1["time_s"], telemetry_q_kpa, label="Telemetry Stage 1", lw=2, alpha=0.7)
    ax_q.plot(sim_booster["time_s"], sim_booster["dynamic_pressure_kpa"], label="Sim Booster Return", lw=2, ls="--")
    ax_q.set_ylabel("Dynamic pressure [kPa]")
    ax_q.set_title("Stage 1 & Booster Return Dynamic Pressure")
    ax_q.grid(True)
    ax_q.legend()

    ax_heat.plot(sim_stage1["time_s"], sim_stage1["heat_flux_kw_m2"], label="Sim Stage 1", lw=2)
    ax_heat.plot(tel_stage1["time_s"], telemetry_heat_kw_m2, label="Telemetry Stage 1 (estimated)", lw=2, alpha=0.7)
    ax_heat.plot(sim_booster["time_s"], sim_booster["heat_flux_kw_m2"], label="Sim Booster Return", lw=2, ls="--")
    ax_heat.set_ylabel("Heat flux [kW/m²]")
    ax_heat.set_xlabel("Time since liftoff [s]")
    ax_heat.set_title("Stage 1 & Booster Return Empirical Heat Flux")
    ax_heat.grid(True)
    ax_heat.legend()

    fig.tight_layout()


def main():
    telemetry = load_telemetry(TELEMETRY_FILE)
    lvopt, solution = run_nominal_simulation()
    sim_df = simulation_profile_to_df(solution, lvopt)

    tel_df = telemetry.copy()
    tel_df["time_s"] = tel_df["time_s"] - tel_df["time_s"].min()

    # Align sim time so its initial velocity matches the first telemetry velocity
    sim_df, time_shift = align_sim_time_by_velocity(sim_df, tel_df)

    # Drop negative times (if any)
    tel_df = tel_df[tel_df["time_s"] >= 0.0]

    compare_and_plot(sim_df, tel_df)

    # Plot simulated second stage
    sim_stage2 = simulation_stage2_profile(solution, lvopt, time_shift=time_shift-0)
    if not sim_stage2.empty:
        tel_stage2 = tel_df[
            (tel_df["time_s"] >= sim_stage2["time_s"].min() - 5.0)
            & (tel_df["time_s"] <= sim_stage2["time_s"].max()+120)
        ]
        if not tel_stage2.empty:
            # use stage2-specific telemetry columns
            tel_stage2 = tel_stage2[["time_s", "altitude2_km", "velocity2_kmh", "acceleration_g"]].rename(
                columns={
                    "altitude2_km": "altitude_km",
                    "velocity2_kmh": "velocity_kmh",
                    "acceleration_g": "accel_g",
                }
            )
            compare_and_plot(sim_stage2, tel_stage2, stage_label="Second Stage")




    # Plot simulated booster return (if available)
    sim_booster = simulation_booster_profile(solution, lvopt, time_shift=time_shift)
    if not sim_booster.empty:
        plot_stage1_and_booster(
            sim_df, tel_df, sim_booster,
            rho_fun=lvopt.atmosphere.rho_fun,
            heat_flux_coefficient=lvopt.booster.Booster_k_empirical,
        )
        
    plt.show()


if __name__ == "__main__":
    main()
