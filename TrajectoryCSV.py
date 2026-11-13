"""Wide CSV flight histories at original timestamps (no resampling)."""
from dataclasses import asdict
import csv
import json
import re

import numpy as np
import pandas as pd

from IntegratedResults import second_stage_coast, return_controls
from LV_Type_scaled import VLType, DispesrionFactorsType
from TelemetryCSV import aerodynamic_loads, numerical_acceleration
from TelemetryFrames import eci_to_ecef_velocity
from Trajectory3D import second_burn_history
from VehicleDefinitions import vehicle_name


# TelemetryData stores these channels in plotting units; export headers state them.
TELEMETRY_UNITS = {
    'altitude': 'km', 'speed': 'mps', 'acceleration': 'mps2', 'mass': 't',
    'thrust_factor': 'fraction', 'alpha': 'deg', 'pressure': 'kpa',
    'heat_flux': 'kw_m2', 'isp': 's', 'vx': 'mps', 'vy': 'mps', 'vz': 'mps',
    'flight_path': 'deg', 'thrust': 'kn', 'inclination': 'deg', 'apogee': 'km',
    'perigee': 'km', 'thrust_angle': 'deg', 'thrust_x': 'kn', 'thrust_y': 'kn',
    'thrust_z': 'kn', 'radius': 'km', 'radial': 'mps', 'transverse': 'mps',
    'energy': 'mj_kg', 'downrange': 'km', 'local_z': 'km', 'eccentricity': 'dimensionless',
}


def _frame(time, columns, prefix):
    """Occurrence index preserves multiple solver states at a phase boundary."""
    time = np.asarray(time, dtype=float).ravel()
    if not np.isfinite(time).all():
        raise ValueError('Export timestamps must be finite.')
    frame = pd.DataFrame(index=np.arange(len(time)))
    for name, values in columns.items():
        values = np.asarray(values).ravel()
        if len(values) not in (len(time), len(time) - 1):
            raise ValueError(f'{prefix}.{name}: data length does not match time.')
        # Controls/specific acceleration have no sample at the last state.
        frame[f'{prefix}.{name}'] = pd.Series(values)
    frame.index = pd.MultiIndex.from_arrays(
        [time, pd.Series(time).groupby(time, sort=False).cumcount()],
        names=['time_since_launch_s', 'occurrence'])
    return frame


def _state_columns(time, state, altitude, local, density):
    state = np.asarray(state, dtype=float)
    velocity = state[2:4] if local else state[3:6]
    relative = velocity if local else eci_to_ecef_velocity(state[:3], velocity, time)
    speed = np.linalg.norm(relative, axis=0)
    columns = {'altitude_m': np.ravel(altitude), 'mass_kg': state[4 if local else 6],
               'speed_ecef_mps': speed,
               'speed_acceleration_ecef_mps2': numerical_acceleration(time, speed)}
    components = (('position_local_x_m', 'position_local_z_m', 'velocity_local_x_mps',
                   'velocity_local_z_mps', 'mass_kg') if local else
                  ('position_eci_x_m', 'position_eci_y_m', 'position_eci_z_m',
                   'velocity_eci_x_mps', 'velocity_eci_y_mps', 'velocity_eci_z_mps', 'mass_kg'))
    columns.update(zip(components, state))
    if not local:
        columns['speed_eci_mps'] = np.linalg.norm(velocity, axis=0)
        columns.update((f'velocity_ecef_{axis}_mps', value) for axis, value in zip('xyz', relative))
    loads = aerodynamic_loads(np.ravel(altitude), speed, density_factor=density)
    columns['Qdyn_derived_pa'] = loads['pressure'] * 1000
    columns['heat_flux_derived_kw_m2'] = loads['heat_flux']
    return columns


def case_frames(name, solution, target=None):
    density = solution.get('dispersion', {}).get('AtmosphereDensity', 1.)
    for number, phase in ((1, 'stage1'), (2, 'stage2_before_fairing'),
                          (3, 'stage2_after_fairing'), (4, 'booster')):
        if f't{number}_vec' not in solution:
            continue
        time = np.ravel(solution[f't{number}_vec'])
        state = np.asarray(solution[f'x{number}'])
        if number == 4:
            state = state.T
        columns = _state_columns(time, state, solution[f'alt{number}_sol'], number in (1, 4), density)
        # Include every saved scalar history for this phase, including future fields.
        labels = {'alt': 'altitude_m', 'Qdyn': 'Qdyn_pa', 'acc': 'specific_acceleration_mps2',
                  'Isp': 'isp_s', 'i': 'inclination_rad', 'Alpha': 'alpha_rad',
                  'apogee': 'apogee_radius_m', 'perigee': 'perigee_radius_m'}
        for key, values in solution.items():
            match = re.fullmatch(r'(.+)' + str(number) + r'_sol', key)
            if match:
                columns[labels.get(match[1], key)] = np.ravel(values)
        if f'u{number}' in solution and number != 4:
            controls = np.asarray(solution[f'u{number}'])
            labels = ('thrust_factor', 'angle_of_attack_rad') if number == 1 else (
                'thrust_eci_x_n', 'thrust_eci_y_n', 'thrust_eci_z_n')
            columns.update(zip(labels, controls))
            if number != 1:
                columns['thrust_n'] = np.linalg.norm(controls, axis=0)
        yield _frame(time, columns, f'{name}.{phase}')

    if all(key in solution for key in ('t1_vec', 't2_vec', 'x1', 'LaunchAz', 'payload_mass')):
        coast = second_stage_coast(solution)
        if coast is not None:
            time, state, altitude = coast
            yield _frame(time, _state_columns(time, state, altitude, False, density), f'{name}.stage2_coast')
    if 'x4' in solution and 'dt4_landing' in solution:
        model = VLType()
        model.ApplyVehicleConfiguration(solution.get('configuration', 1))
        model.ApplyScenarioDispersionToLV(DispesrionFactorsType(**solution.get('dispersion', {})))
        time, throttle, angle = return_controls(solution, model)
        yield _frame(time, {'thrust_factor': throttle, 'thrust_velocity_angle_deg': angle},
                     f'{name}.booster_controls')
    burn = second_burn_history(solution, target)
    if burn is not None:
        # The plot uses an illustrative offset. CSV always uses physical launch time.
        time = burn['physical_start'] + burn['time'] - burn['time'][0]
        columns = _state_columns(time, burn['state'], burn['altitude'], False, density)
        columns.update((f'thrust_eci_{axis}_n', value) for axis, value in zip('xyz', burn['thrust']))
        columns['isp_s'] = np.full(len(time), burn['isp'])
        yield _frame(time, columns, f'{name}.stage2_burn2_illustration')


def case_metadata(solution, target=None, request=None):
    """Describe saved run inputs, preferring result metadata over its request."""
    saved = dict(request or {})
    saved.update(solution)
    columns = {'payload_mass_kg': float(saved.get('payload_mass', np.nan))}
    configuration = saved.get('configuration')
    if configuration is not None:
        columns['launch_vehicle_name'] = vehicle_name(configuration)
    site = saved.get('launch_site') or {}
    for key, label in (('name', 'name'), ('latitude', 'latitude_deg'),
                       ('longitude', 'longitude_deg'), ('altitude', 'altitude_m')):
        columns[f'launch_site_{label}'] = site.get(key, np.nan)
    orbit = target if target is not None else saved.get('target_orbit', {})
    orbit = orbit or {}
    columns['target_orbit_name'] = orbit.get('Name', np.nan)
    for key in ('apogee', 'perigee'):
        radius = orbit.get(key, np.nan)
        columns[f'target_{key}_radius_m'] = radius
        columns[f'target_{key}_altitude_km'] = (float(radius) - VLType.R0) / 1000.
    columns['target_inclination_deg'] = float(np.rad2deg(orbit.get('i', np.nan)))
    nominal = asdict(DispesrionFactorsType())
    dispersions = dict((request or {}).get('dispersion') or {})
    dispersions.update(solution.get('dispersion') or {})
    changed = {key: float(value) for key, value in dispersions.items()
               if key not in nominal or float(value) != nominal[key]}
    # An explicit empty object distinguishes a nominal run from missing metadata.
    columns['non_nominal_dispersions_json'] = (json.dumps(changed, sort_keys=True)
                                               if 'dispersion' in saved else np.nan)
    columns.update((f'dispersion.{key}', value) for key, value in changed.items())
    return columns


def export_trajectory_csv(path, cases=(), *, telemetry=None, telemetry_shift=0.,
                          primary_group='Stage 1 / booster', case_requests=None):
    """Export (name, solution, target) tuples with optional saved requests by name."""
    frames = []
    metadata = []
    for name, solution, target in cases:
        frames.extend(case_frames(name, solution, target))
        request = (case_requests or {}).get(name)
        metadata.extend((f'# {name}', key, '' if pd.isna(value) else value)
                        for key, value in case_metadata(solution, target, request).items())
    if telemetry is not None:
        time = np.asarray(telemetry.time) + telemetry_shift
        raw = getattr(telemetry, 'raw_columns', None)
        if raw:
            frames.append(_frame(time, raw, 'telemetry.original'))
        for group in telemetry.groups:
            channels = (telemetry.groups[group] if group == 'primary' else
                        telemetry.channels(group, primary_group))
            columns = {}
            for key, values in channels.items():
                base = key.removesuffix('_derived')
                unit = TELEMETRY_UNITS.get(base, 'dimensionless')
                columns[f'{key}_{unit}'] = values
            if columns:
                frames.append(_frame(time, columns, f'telemetry.{group}'))
    if not frames:
        raise ValueError('Select at least one completed case or loaded telemetry dataset.')
    result = pd.concat(frames, axis=1).sort_index(kind='stable')
    result = result.reset_index(level='occurrence', drop=True).reset_index()
    with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
        if metadata:
            writer = csv.writer(stream)
            writer.writerow(['# Case', 'Parameter', 'Value'])
            writer.writerows(metadata)
            writer.writerow([])
        result.to_csv(stream, index=False, na_rep='')
    return len(result), len(result.columns)
