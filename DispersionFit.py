"""Bounded dispersion fitting against synchronized telemetry, independent of Tk.

Each successful candidate is synchronized before scoring. The trajectory solver
is injectable for testing; by default each evaluation runs the full optimizer.
"""
from copy import deepcopy
from dataclasses import asdict, fields
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.optimize import minimize

from LV_Type_scaled import DispesrionFactorsType, VLType
from LaunchSites import parse_launch_site
from TelemetryCSV import (TelemetryData, load_telemetry_csv, interpolate_speed_time,
                          numerical_acceleration, aerodynamic_loads, PRIMARY_GROUPS)
from TelemetryFrames import eci_to_ecef_velocity

DEFAULT_PARAMETERS = ('FirstStageThrust', 'FirstStageIsp', 'SecondStageThrust', 'SecondStageIsp','FirstStageCx0', 'BoosterStageCx0','FirstStage_EmptyMass','SecondStage_EmptyMass','FirstStage_PropMass','SecondStage_PropMass')
MULTIPLIERS = {'FirstStageThrust', 'FirstStageIsp', 'SecondStageThrust', 'SecondStageIsp',
               'FirstStageCx0', 'BoosterStageCx0', 'AtmosphereDensity'}
ERROR_SCALES = {'altitude_scale': 1000., 'speed_scale': 25.,
                'acceleration_scale': 1., 'pressure_scale': 5., 'phase_time_scale': 5.}
CHANNEL_UNITS = {'altitude': 'm', 'speed': 'm/s',
                 'acceleration': 'm/s²', 'pressure': 'kPa'}


def default_fit_bounds(request):
    """Nominal bounds for the vehicle, independent of the starting factors."""
    model = VLType()
    model.ApplyVehicleConfiguration(request.get('configuration', 1))
    values = asdict(DispesrionFactorsType())
    widths = {'FirstStage_EmptyMass': .1 * model.EmptyFirstStageMass,
              'SecondStage_EmptyMass': .1 * model.EmptySecondStageMass,
              'FirstStage_PropMass': .02 * model.FirstStagePropellentMass,
              'SecondStage_PropMass': .02 * model.SecondStagePropellentMass,
              'LaunchAltDelta': 100., 'StagePartitionDelta': .02}
    result = {}
    for name, value in values.items():
        value = float(value)
        width = (.1 if name.endswith('Isp') else .15) * value if name in MULTIPLIERS else widths[name]
        result[name] = (value - width, value + width)
    return result


def _validate_request(request):
    dispersion = DispesrionFactorsType(**request['dispersion'])
    for name, value in asdict(dispersion).items():
        if not np.isfinite(value) or name in MULTIPLIERS and value <= 0:
            raise ValueError(f'Invalid dispersion value: {name}')
    model = VLType()
    model.ApplyVehicleConfiguration(request['configuration'])
    partition = model.FirstStagePropellentMass / model.TotalPropellentMass + dispersion.StagePartitionDelta
    if not 0 < partition < 1:
        raise ValueError('Stage partition must leave propellant in both stages.')
    model.ApplyLaunchSite(request.get('launch_site'))
    model.ApplyScenarioDispersionToLV(dispersion)
    if min(model.EmptyFirstStageMass, model.EmptySecondStageMass,
           model.FirstStagePropellentMass, model.SecondStagePropellentMass,
           model.R0 + model.LaunchAltitude) <= 0:
        raise ValueError('Dispersion values produce invalid mass or launch altitude.')
    return dispersion


def _solve_candidate(request, init_guess=None, ipopt_options=None):
    from LV_Optimization_Type import LV_Optimization
    dispersion = _validate_request(request)
    model = LV_Optimization(dispersion, request['configuration'],
                            launch_site=parse_launch_site(request.get('launch_site')))
    if init_guess is not None:
        compatible = (init_guess.get('configuration', request['configuration']) == request['configuration']
                      and parse_launch_site(init_guess.get('launch_site')) == parse_launch_site(request.get('launch_site')))
        if request['recovery'] != 'EXP' and 'x4' not in init_guess:
            compatible = False
        if request['recovery'] == 'RTLS' and 'x4_boostback' not in init_guess:
            compatible = False
        if not compatible:
            init_guess = None
    return model.SolveOptimiztion(
        RecoveryStrategy=request['recovery'], Target_Orbit=request['target_orbit'],
        payload_mass_predefined=request['payload_mass_predefined'], init_guess=init_guess,
        Plot_interm=False, print_ipopt=False, output_dir=None, ipopt_options=ipopt_options)


def simulation_channels(solution, phase):
    """Earth-fixed kinematics, with acceleration/pressure for ascent only."""
    def specific_acceleration(number, time):
        values = np.asarray(solution.get(f'acc{number}_sol', []), dtype=float).ravel()
        if len(values) == len(time):
            return values
        if len(values) == len(time) - 1:
            # Controls are defined at the first N states, not the terminal state.
            return np.r_[values, np.nan]
        if not len(values):
            return np.full(len(time), np.nan)
        raise ValueError(f'Incomplete stage {number} specific acceleration.')

    if phase == 'stage1':
        t = np.asarray(solution['t1_vec']).ravel()
        altitude = np.asarray(solution['alt1_sol']).ravel()
        speed = np.linalg.norm(np.asarray(solution['x1'])[2:4], axis=0)
        acceleration = specific_acceleration(1, t)
    elif phase == 'booster':
        if not all(key in solution for key in ('t4_vec', 'alt4_sol', 'x4')):
            raise ValueError('Incomplete booster simulation.')
        t = np.asarray(solution['t4_vec']).ravel()
        altitude = np.asarray(solution['alt4_sol']).ravel()
        speed = np.linalg.norm(np.asarray(solution['x4'])[:, 2:4], axis=1)
    else:
        times, altitudes, speeds, accelerations = [], [], [], []
        for number in (2, 3):
            tn = np.asarray(solution[f't{number}_vec']).ravel()
            state = np.asarray(solution[f'x{number}'])
            times.append(tn)
            altitudes.append(np.asarray(solution[f'alt{number}_sol']).ravel())
            speeds.append(np.linalg.norm(eci_to_ecef_velocity(state[:3], state[3:6], tn), axis=0))
            accelerations.append(specific_acceleration(number, tn))
        t, altitude, speed = map(np.concatenate, (times, altitudes, speeds))
        acceleration = np.concatenate(accelerations)
    if not (len(t) == len(altitude) == len(speed)) or len(t) < 2:
        raise ValueError(f'Incomplete {phase} simulation.')
    if not np.isfinite(np.concatenate((t, altitude, speed))).all() or np.any(np.diff(t) < 0):
        raise ValueError(f'Invalid {phase} simulation samples or decreasing time.')
    # Adjacent burns share a boundary timestamp; do not create a zero interval.
    unique, indices = np.unique(t, return_index=True)
    if phase == 'booster':
        return unique, {'altitude': altitude[indices], 'speed': speed[indices]}
    # Use atmosphere-relative speed for every phase, including ECI stage two.
    density = solution.get('dispersion', {}).get('AtmosphereDensity', 1.)
    pressure = aerodynamic_loads(altitude[indices], speed[indices], density)['pressure']
    # At a shared burn boundary use the acceleration from the following burn
    # when the previous burn has no terminal control value.
    specific = acceleration[indices].copy()
    for index in np.flatnonzero(~np.isfinite(specific)):
        matches = acceleration[t == unique[index]]
        finite = matches[np.isfinite(matches)]
        if len(finite):
            specific[index] = finite[0]
    return unique, {'altitude': altitude[indices], 'speed': speed[indices],
                    'specific_acceleration': specific, 'pressure': pressure}


def _fit_telemetry_channels(telemetry, phase, primary_group):
    """Measured values take precedence; calculate missing acceleration/pressure.

    Supplied acceleration follows the CSV convention of specific acceleration.
    Its fallback is signed Earth-fixed dv/dt, compared to the same speed interval
    in each candidate rather than to the simulator's proper acceleration.
    """
    channels = telemetry.channels(phase, primary_group)
    if phase not in ('stage1', 'stage2'):
        return channels, {}
    calculated = {}
    if 'speed' in channels:
        calculated['acceleration'] = numerical_acceleration(telemetry.time, channels['speed'])
    if 'altitude' in channels and 'speed' in channels:
        calculated['pressure'] = aerodynamic_loads(channels['altitude'] * 1000,
                                                   channels['speed'])['pressure']
    measured = {}
    for channel in ('acceleration', 'pressure'):
        supplied = np.asarray(channels.get(channel, np.full(len(telemetry.time), np.nan)))
        measured[channel] = np.isfinite(supplied) & (supplied >= 0)
        channels[channel] = np.where(measured[channel], supplied,
                                     calculated.get(channel, np.nan))
    return channels, measured


def _sample_prediction(sample, simulation, shift):
    time, channels = simulation
    times = sample['time'] + shift
    channel = sample['channel']
    if channel != 'acceleration':
        return np.interp(times, time, channels[channel], left=np.nan, right=np.nan)
    result = np.full(len(times), np.nan)
    measured = sample['measured']
    if measured.any():
        result[measured] = np.interp(times[measured], time, channels['specific_acceleration'],
                                     left=np.nan, right=np.nan)
    calculated = ~measured
    if calculated.any():
        previous = sample['previous_time'][calculated] + shift
        speed = np.interp(times[calculated], time, channels['speed'], left=np.nan, right=np.nan)
        prior_speed = np.interp(previous, time, channels['speed'], left=np.nan, right=np.nan)
        result[calculated] = (speed - prior_speed) / (times[calculated] - previous)
    return result


def synchronize_candidate(solution, telemetry, primary_group='Stage 1 / booster'):
    """Align first simulated speed to its first rising telemetry crossing.

    Prefer stage one, as the GUI Sync button does. If only stage-two speed is
    available, use its first simulated point. Never accumulate previous shifts.
    """
    for phase in ('stage1', 'stage2'):
        channels = telemetry.channels(phase, primary_group)
        if 'speed' not in channels:
            continue
        t, sim = simulation_channels(solution, phase)
        matched = interpolate_speed_time(telemetry.time, channels['speed'], float(sim['speed'][0]))
        return {'shift': float(t[0] - matched), 'phase': phase,
                'simulation_time': float(t[0]), 'telemetry_time': float(matched),
                'speed_mps': float(sim['speed'][0])}
    raise ValueError('Fitting needs first- or second-stage speed telemetry for synchronization.')


def _phase_telemetry_bounds(solution, telemetry, phase, primary_group, synchronization):
    """Full observed phase extent, independent of fit margins and acceleration.

    Unlabelled primary telemetry shared by ascent and return is split at the
    baseline separation time. Freeze these bounds with the comparison samples
    so later candidates cannot move the telemetry target with their own timing.
    Explicit phase columns retain their complete observed extent.
    """
    channels = telemetry.channels(phase, primary_group)
    keys = ('altitude', 'speed') if phase == 'booster' else CHANNEL_UNITS
    present = np.zeros(len(telemetry.time), dtype=bool)
    explicit = present.copy()
    for key in keys:
        if key in channels:
            valid = np.isfinite(channels[key])
            if key != 'altitude':
                valid &= channels[key] >= 0
            present |= valid
        if key in telemetry.groups[phase]:
            explicit |= np.isfinite(telemetry.groups[phase][key])
    if (primary_group == 'Stage 1 / booster' and phase in ('stage1', 'booster')
            and 't4_vec' in solution):
        separation = float(np.asarray(solution['t4_vec']).ravel()[0]) - synchronization['shift']
        in_phase = telemetry.time <= separation if phase == 'stage1' else telemetry.time >= separation
        present &= explicit | in_phase
    time = telemetry.time[present & np.isfinite(telemetry.time)]
    if not len(time):
        return None
    return float(time.min()), float(time.max())


def _comparison_samples(solution, telemetry, primary_group, synchronization, windows=None, *, recovery='EXP'):
    samples = []
    phases = ('stage1', 'stage2') if recovery == 'EXP' else ('stage1', 'stage2', 'booster')
    for phase in phases:
        channels, measured = _fit_telemetry_channels(telemetry, phase, primary_group)
        fit_channels = ('altitude', 'speed') if phase == 'booster' else CHANNEL_UNITS
        if phase == 'booster':
            if not any(key in channels and np.isfinite(channels[key]).any() for key in fit_channels):
                continue
            # Return burns decelerate the booster. Fit the entire return history,
            # including coasts, and allow independently supplied channels.
            eligible = np.ones(len(telemetry.time), dtype=bool)
        else:
            if 'speed' not in channels:
                continue
            # Use signed ECEF speed acceleration for both ascent stages, rather
            # than an accelerometer's thrust/proper acceleration. Keep gaps.
            eligible = numerical_acceleration(telemetry.time, channels['speed']) > 0
            if not eligible.any():
                continue
        simulation = simulation_channels(solution, phase)
        phase_bounds = _phase_telemetry_bounds(solution, telemetry, phase, primary_group, synchronization)
        t, _ = simulation
        # Fix the sample set once, using baseline overlap. A small margin avoids
        # making a slightly different synchronization fail at the first sample.
        margin = min(2., .05 * (t[-1] - t[0]))
        low, high = t[0] + margin, t[-1] - margin
        if windows and phase in windows:
            low, high = windows[phase]
            if phase == 'booster':
                low, high = max(low, t[0]), min(high, t[-1])
        shifted_time = telemetry.time + synchronization['shift']
        for channel in fit_channels:
            if channel not in channels:
                continue
            values = np.asarray(channels[channel]) * (1000 if channel == 'altitude' else 1)
            valid = (eligible & np.isfinite(values) & np.isfinite(shifted_time)
                     & (shifted_time >= low) & (shifted_time <= high))
            if channel == 'speed':
                valid &= values >= 0
            raw_t, observations = telemetry.time[valid], values[valid]
            if len(raw_t) < 3:
                continue
            # Repeated mission-clock values do not get disproportionate weight.
            unique, inverse, count = np.unique(raw_t, return_inverse=True, return_counts=True)
            observations = np.bincount(inverse, weights=observations) / count
            sample = {'phase': phase, 'channel': channel, 'time': unique,
                      'values': observations, 'phase_time_bounds': phase_bounds}
            if channel in measured:
                supplied = measured[channel][valid]
                supplied_count = np.bincount(inverse, weights=supplied, minlength=len(unique))
                sample['measured'] = supplied_count > 0
                # At duplicate timestamps prefer measured values over a fallback
                # with a different acceleration definition.
                sums = np.bincount(inverse, weights=np.where(supplied, values[valid], 0))
                np.divide(sums, supplied_count, out=sample['values'], where=sample['measured'])
            if channel == 'acceleration':
                previous = np.r_[np.nan, telemetry.time[:-1]][valid]
                prior_sums = np.bincount(inverse, weights=np.where(supplied, 0, previous))
                sample['previous_time'] = np.full(len(unique), np.nan)
                np.divide(prior_sums, count - supplied_count, out=sample['previous_time'],
                          where=~sample['measured'])
                predicted = _sample_prediction(sample, simulation, synchronization['shift'])
                if sample['measured'].any() and not np.isfinite(predicted[sample['measured']]).any():
                    raise ValueError(f'{phase} simulation lacks specific acceleration for measured telemetry.')
                # Fix intervals only where the baseline has both speed endpoints
                # or a supplied proper-acceleration prediction. Never reselect
                # them for later candidates.
                finite = np.isfinite(predicted)
                for key in ('time', 'values', 'measured', 'previous_time'):
                    sample[key] = sample[key][finite]
            if len(sample['time']) >= 3:
                samples.append(sample)
    if not samples:
        raise ValueError('No overlapping positive-acceleration first/second-stage telemetry'
                         + (' or booster return telemetry' if recovery != 'EXP' else '')
                         + ' with at least three distinct times.')
    return samples


def score_candidate(solution, telemetry, samples, primary_group='Stage 1 / booster',
                    altitude_scale=ERROR_SCALES['altitude_scale'], speed_scale=ERROR_SCALES['speed_scale'],
                    acceleration_scale=ERROR_SCALES['acceleration_scale'], pressure_scale=ERROR_SCALES['pressure_scale'],
                    phase_time_scale=ERROR_SCALES['phase_time_scale']):
    # Synchronization is deliberately performed inside EVERY scoring call.
    synchronization = synchronize_candidate(solution, telemetry, primary_group)
    errors, losses = {}, {}
    scales = {'altitude': altitude_scale, 'speed': speed_scale,
              'acceleration': acceleration_scale, 'pressure': pressure_scale}
    phases, bounds = {}, {}
    for sample in samples:
        phase, channel = sample['phase'], sample['channel']
        # Empty telemetry rows contribute neither error nor missing-coverage cost.
        observed = np.isfinite(sample['time']) & np.isfinite(sample['values'])
        if not observed.any():
            continue
        if phase not in phases:
            phases[phase] = simulation_channels(solution, phase)
            bounds[phase] = sample.get('phase_time_bounds')
            if bounds[phase] is None:
                bounds[phase] = _phase_telemetry_bounds(solution, telemetry, phase, primary_group, synchronization)
        interpolated = _sample_prediction(sample, phases[phase], synchronization['shift'])
        valid = observed & np.isfinite(interpolated)
        coverage = float(valid.sum() / observed.sum())
        if coverage < .9:
            raise ValueError(f'{phase} candidate covers less than 90% of the fixed telemetry window.')
        residual = interpolated[valid] - sample['values'][valid]
        normalized = residual / scales[channel]
        robust = 2 * (np.hypot(1., normalized) - 1.)
        # Missing coverage has a large fixed penalty, never a reduced denominator.
        losses.setdefault(phase, []).append(float((robust.sum() + 10000 * (observed & ~valid).sum()) / observed.sum()))
        key = f'{phase}.{channel}'
        errors[key] = {'rmse': float(np.sqrt(np.mean(residual**2))),
                       'unit': CHANNEL_UNITS[channel],
                       'samples': int(valid.sum()), 'coverage': coverage}
        if 'measured' in sample:
            errors[key].update(measured_samples=int((sample['measured'] & valid).sum()),
                               calculated_samples=int((~sample['measured'] & valid).sum()))
    if not losses:
        raise ValueError('No finite telemetry measurements to score.')
    timing = {}
    for phase, (time, _) in phases.items():
        if bounds[phase] is None:
            continue
        start, end = bounds[phase]
        simulated_duration = float(time[-1] - time[0])
        telemetry_duration = end - start
        duration_error = simulated_duration - telemetry_duration
        lateness = np.maximum(time - synchronization['shift'] - end, 0.)
        duration_penalty = float(2 * (np.hypot(1., duration_error / phase_time_scale) - 1.))
        late_penalty = float(np.mean(2 * (np.hypot(1., lateness / phase_time_scale) - 1.)))
        timing[phase] = {'telemetry_start': start, 'telemetry_end': end,
                         'telemetry_duration': telemetry_duration,
                         'simulation_duration': simulated_duration,
                         'duration_error': duration_error, 'unit': 's',
                         'late_samples': int((lateness > 0).sum()),
                         'end_overrun': float(lateness[-1]),
                         'duration_penalty': duration_penalty, 'late_penalty': late_penalty,
                         'penalty': duration_penalty + late_penalty}
    measurement_score = float(np.mean([np.mean(values) for values in losses.values()]))
    timing_score = float(np.mean([timing[phase]['penalty'] if phase in timing else 0. for phase in losses]))
    return {'score': measurement_score + timing_score,
            'measurement_score': measurement_score, 'timing_score': timing_score,
            'phase_timing': timing, 'errors': errors, 'synchronization': synchronization}


class _StopSearch(Exception):
    pass


def fit_dispersion_to_telemetry(base_request, telemetry, *, parameters=DEFAULT_PARAMETERS,
                                bounds=None, primary_group='Stage 1 / booster', max_evaluations=80,
                                altitude_scale=ERROR_SCALES['altitude_scale'], speed_scale=ERROR_SCALES['speed_scale'],
                                acceleration_scale=ERROR_SCALES['acceleration_scale'],
                                pressure_scale=ERROR_SCALES['pressure_scale'],
                                phase_time_scale=ERROR_SCALES['phase_time_scale'], regularization=.01,
                                initial_solution=None, windows=None, progress_callback=None,
                                checkpoint_callback=None, cancel_callback=None,
                                solver=None, ipopt_options=None):
    """Return the best successful fit, its trajectory, baseline, and diagnostics.

    max_evaluations counts all solver calls, including optional payload bootstrap.
    A nonpositive payload is fixed from initial_solution, or from one preliminary
    nominal solve. All subsequent candidates use that SAME fixed payload.
    Bounds contain absolute factor/offset values. Progress and checkpoint hooks
    run after evaluations; cancellation is checked between solver calls.
    Fit altitude, speed, acceleration and dynamic pressure during ascent.
    Non-EXP recovery includes post-separation booster altitude/speed only,
    without the positive-acceleration filter used for ascent. All phases share
    the first-/second-stage synchronization and fixed baseline sample windows.
    Missing measurements are skipped. Each observed phase also pays for its
    duration mismatch and simulated samples later than its last telemetry point.
    """
    telemetry = load_telemetry_csv(telemetry) if isinstance(telemetry, (str, Path)) else telemetry
    request = deepcopy(base_request)
    request['dispersion'] = dict(asdict(DispesrionFactorsType()), **request['dispersion'])
    _validate_request(request)
    parameters = tuple(parameters)
    known = {item.name for item in fields(DispesrionFactorsType)}
    if not parameters or len(set(parameters)) != len(parameters) or not set(parameters) <= known:
        raise ValueError('Select one or more unique dispersion factors.')
    if type(max_evaluations) is not int or max_evaluations < 2:
        raise ValueError('Evaluation budget must be an integer of at least 2.')
    scales = dict(altitude_scale=altitude_scale, speed_scale=speed_scale,
                  acceleration_scale=acceleration_scale, pressure_scale=pressure_scale,
                  phase_time_scale=phase_time_scale)
    if not np.isfinite([*scales.values(), regularization]).all() or min(scales.values()) <= 0 or regularization < 0:
        raise ValueError('Error scales must be positive; regularization must be nonnegative.')
    if primary_group not in PRIMARY_GROUPS:
        raise ValueError('Unknown telemetry column assignment.')
    if not any('speed' in telemetry.channels(phase, primary_group) for phase in ('stage1', 'stage2')):
        raise ValueError('Load first- or second-stage speed telemetry before fitting.')
    ranges = default_fit_bounds(request) if bounds is None else bounds
    try:
        limits = np.array([ranges[name] for name in parameters], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError('Provide numeric lower/upper bounds for every selected factor.') from None
    start = np.array([request['dispersion'][name] for name in parameters], dtype=float)
    if limits.shape != (len(parameters), 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Every selected factor needs finite lower < upper bounds.')
    if np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
        raise ValueError('The current factor values must be inside their fitting bounds.')
    for index, name in enumerate(parameters):
        if name in MULTIPLIERS and limits[index, 0] <= 0:
            raise ValueError(f'{name}: multiplier bounds must be positive.')
    solver = solver or _solve_candidate
    options = dict(max_iter=300, max_cpu_time=60., print_level=0)
    options.update(ipopt_options or {})
    span = limits[:, 1] - limits[:, 0]
    initial = (start - limits[:, 0]) / span
    history, cache = [], {}
    best = baseline = None
    samples = None
    warm = initial_solution
    calls = 0
    started = time.monotonic()
    termination = 'Evaluation budget reached'

    def stopped():
        if cancel_callback and cancel_callback():
            raise _StopSearch('Cancelled')
        if calls >= max_evaluations:
            raise _StopSearch('Evaluation budget reached')

    def publish(record):
        history.append(record)
        if progress_callback:
            progress_callback(dict(record, best_score=best['score'] if best else None,
                                   max_evaluations=max_evaluations, elapsed=time.monotonic() - started))

    payload = float(request['payload_mass_predefined'])
    if not np.isfinite(payload):
        raise ValueError('Payload mass must be finite.')
    if payload <= 0:
        if initial_solution is not None and initial_solution.get('success'):
            payload = float(initial_solution['payload_mass'])
        else:
            stopped()
            calls += 1
            warm = solver(request, init_guess=None, ipopt_options=options)
            if not warm.get('success'):
                raise ValueError('The preliminary payload optimization failed. Set a fixed positive payload and retry.')
            payload = float(warm['payload_mass'])
            publish({'evaluation': calls, 'kind': 'payload', 'payload_mass': payload,
                     'score': None, 'status': 'Payload fixed from nominal optimization'})
        if not np.isfinite(payload) or payload <= 0:
            raise ValueError('Fitting requires a positive fixed payload mass.')
        request['payload_mass_predefined'] = payload

    def evaluate(normalized):
        nonlocal calls, warm, best, baseline, samples
        stopped()
        values = limits[:, 0] + np.clip(normalized, 0, 1) * span
        key = tuple(np.round(values, 12))
        if key in cache:
            return cache[key]
        candidate_request = deepcopy(request)
        candidate_request['dispersion'].update(zip(parameters, map(float, values)))
        calls += 1
        record = {'evaluation': calls, 'kind': 'candidate', 'parameters': dict(zip(parameters, map(float, values)))}
        try:
            _validate_request(candidate_request)
            solution = solver(candidate_request, init_guess=warm, ipopt_options=options)
            if not solution.get('success'):
                raise ValueError(f"Solver did not converge: {solution.get('return_status', 'unknown')}")
            if samples is None:
                synchronization = synchronize_candidate(solution, telemetry, primary_group)
                samples = _comparison_samples(solution, telemetry, primary_group, synchronization,
                                              windows, recovery=request['recovery'])
            metrics = score_candidate(solution, telemetry, samples, primary_group, **scales)
            penalty = float(regularization * np.mean((normalized - initial)**2))
            score = metrics['score'] + penalty
            candidate = dict(metrics, score=score, data_score=metrics['score'], request=candidate_request,
                             solution=solution, evaluation=calls)
            if baseline is None:
                baseline = candidate
            warm = solution
            record.update(score=score, status='OK', shift=metrics['synchronization']['shift'],
                          errors=metrics['errors'], phase_timing=metrics['phase_timing'])
            if best is None or score < best['score']:
                best = candidate
                if checkpoint_callback:
                    checkpoint_callback({'best': best, 'baseline': baseline, 'history': history + [record],
                                         'evaluations': calls, 'termination': 'Searching',
                                         'parameters': list(parameters), 'bounds': dict(zip(parameters, limits.tolist())),
                                         'settings': dict(scales, regularization=regularization,
                                                          primary_group=primary_group)})
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            score = 1e12
            record.update(score=None, status=str(exc))
        cache[key] = score
        publish(record)
        return score

    try:
        evaluate(initial)
        if baseline is None:
            raise ValueError('Could not score the baseline: ' + history[-1]['status'])
        optimized = minimize(evaluate, initial, method='Powell', bounds=[(0., 1.)] * len(parameters),
                             options={'maxfev': max_evaluations * 4, 'xtol': .002, 'ftol': .001})
        termination = str(optimized.message)
    except _StopSearch as exc:
        termination = str(exc)
    if best is None:
        raise ValueError('Search stopped before any synchronized, successful comparison was available.')
    return {'best': best, 'baseline': baseline, 'history': history, 'evaluations': calls,
            'termination': termination, 'elapsed': time.monotonic() - started,
            'parameters': list(parameters), 'bounds': dict(zip(parameters, limits.tolist())),
            'settings': dict(scales, regularization=regularization, primary_group=primary_group),
            'windows': [{'phase': sample['phase'], 'channel': sample['channel'],
                         'telemetry_start': float(sample['time'][0]), 'telemetry_end': float(sample['time'][-1])}
                        for sample in samples]}


def fit_report(result):
    """JSON-safe diagnostics without the large simulation arrays."""
    report = {key: value for key, value in result.items() if key not in ('best', 'baseline')}
    for name in ('baseline', 'best'):
        report[name] = {key: value for key, value in result[name].items() if key != 'solution'}
    return report


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def _atomic_pickle(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream)
    temporary.replace(path)


def worker(directory):
    directory = Path(directory)
    settings = json.loads((directory / 'fit_inputs.json').read_text())
    data = settings.pop('telemetry')
    telemetry = TelemetryData(Path(data['path']), np.array(data['time']),
                              {group: {key: np.array(value) for key, value in channels.items()}
                               for group, channels in data['groups'].items()}, data['time_note'])
    initial_solution = None
    if (directory / 'initial_solution.pickle').exists():
        # This is a private snapshot written by the GUI for this worker.
        with (directory / 'initial_solution.pickle').open('rb') as stream:
            initial_solution = pickle.load(stream)
    events = []
    def progress(event):
        events.append({key: value for key, value in event.items() if key not in ('best_score', 'max_evaluations', 'elapsed')})
        _atomic_json(directory / 'fit_history.json', events)
        _atomic_json(directory / 'fit_progress.json', event)
        print(f"Fit evaluation {event['evaluation']}/{event['max_evaluations']}: "
              f"score={event.get('score')} best={event.get('best_score')} · {event['status']}", flush=True)
    def checkpoint(result):
        _atomic_pickle(directory / 'fit_best.pickle', result)
        _atomic_json(directory / 'fit_report.json', fit_report(result))
        _atomic_pickle(directory / 'solution.pickle', result['best']['solution'])
        _atomic_json(directory / 'inputs.json', result['best']['request'])
    result = fit_dispersion_to_telemetry(telemetry=telemetry, initial_solution=initial_solution,
                                        progress_callback=progress, checkpoint_callback=checkpoint,
                                        cancel_callback=lambda: (directory / 'CANCEL').exists(), **settings)
    _atomic_pickle(directory / 'fit_result.pickle', result)
    _atomic_json(directory / 'fit_report.json', fit_report(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    worker(parser.parse_args().worker)
