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
from TelemetryCSV import TelemetryData, load_telemetry_csv, interpolate_speed_time, numerical_acceleration
from TelemetryFrames import eci_to_ecef_velocity

DEFAULT_PARAMETERS = ('FirstStageThrust', 'FirstStageIsp', 'SecondStageThrust', 'SecondStageIsp','FirstStageCx0', 'BoosterStageCx0','FirstStage_EmptyMass','SecondStage_EmptyMass','FirstStage_PropMass','SecondStage_PropMass')
MULTIPLIERS = {'FirstStageThrust', 'FirstStageIsp', 'SecondStageThrust', 'SecondStageIsp',
               'FirstStageCx0', 'BoosterStageCx0', 'AtmosphereDensity'}


def default_fit_bounds(request):
    """Nominal bounds for the vehicle, independent of the starting factors."""
    model = VLType()
    model.ApplyVehicleConfiguration(request.get('configuration', 1))
    values = asdict(DispesrionFactorsType())
    widths = {'FirstStage_EmptyMass': .02 * model.EmptyFirstStageMass,
              'SecondStage_EmptyMass': .02 * model.EmptySecondStageMass,
              'FirstStage_PropMass': .02 * model.FirstStagePropellentMass,
              'SecondStage_PropMass': .02 * model.SecondStagePropellentMass,
              'LaunchAltDelta': 100., 'StagePartitionDelta': .02}
    result = {}
    for name, value in values.items():
        value = float(value)
        width = (.05 if name.endswith('Isp') else .15) * value if name in MULTIPLIERS else widths[name]
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
    """Time, altitude in metres, and speed in m/s in the telemetry frame."""
    if phase == 'stage1':
        t = np.asarray(solution['t1_vec']).ravel()
        altitude = np.asarray(solution['alt1_sol']).ravel()
        speed = np.linalg.norm(np.asarray(solution['x1'])[2:4], axis=0)
    elif phase == 'booster':
        if not all(key in solution for key in ('t4_vec', 'alt4_sol', 'x4')):
            raise ValueError('Incomplete booster simulation.')
        t = np.asarray(solution['t4_vec']).ravel()
        altitude = np.asarray(solution['alt4_sol']).ravel()
        speed = np.linalg.norm(np.asarray(solution['x4'])[:, 2:4], axis=1)
    else:
        times, altitudes, speeds = [], [], []
        for number in (2, 3):
            tn = np.asarray(solution[f't{number}_vec']).ravel()
            state = np.asarray(solution[f'x{number}'])
            times.append(tn)
            altitudes.append(np.asarray(solution[f'alt{number}_sol']).ravel())
            speeds.append(np.linalg.norm(eci_to_ecef_velocity(state[:3], state[3:6], tn), axis=0))
        t, altitude, speed = map(np.concatenate, (times, altitudes, speeds))
    if not (len(t) == len(altitude) == len(speed)) or len(t) < 2:
        raise ValueError(f'Incomplete {phase} simulation.')
    if not np.isfinite(np.concatenate((t, altitude, speed))).all() or np.any(np.diff(t) < 0):
        raise ValueError(f'Invalid {phase} simulation samples or decreasing time.')
    # Adjacent burns share a boundary timestamp; do not create a zero interval.
    unique, indices = np.unique(t, return_index=True)
    return unique, {'altitude': altitude[indices], 'speed': speed[indices]}


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


def _comparison_samples(solution, telemetry, primary_group, synchronization, windows=None, *, recovery='EXP'):
    samples = []
    phases = ('stage1', 'stage2') if recovery == 'EXP' else ('stage1', 'stage2', 'booster')
    for phase in phases:
        channels = telemetry.channels(phase, primary_group)
        if phase == 'booster':
            if not any(key in channels for key in ('altitude', 'speed')):
                continue
            # Return burns decelerate the booster. Fit the entire return history,
            # including coasts, and allow independently measured altitude/speed.
            eligible = np.ones(len(telemetry.time), dtype=bool)
        else:
            if 'speed' not in channels:
                continue
            # Use signed ECEF speed acceleration for both ascent stages, rather
            # than an accelerometer's thrust/proper acceleration. Keep gaps.
            eligible = numerical_acceleration(telemetry.time, channels['speed']) > 0
        t, _ = simulation_channels(solution, phase)
        # Fix the sample set once, using baseline overlap. A small margin avoids
        # making a slightly different synchronization fail at the first sample.
        margin = min(2., .05 * (t[-1] - t[0]))
        low, high = t[0] + margin, t[-1] - margin
        if windows and phase in windows:
            low, high = windows[phase]
            if phase == 'booster':
                low, high = max(low, t[0]), min(high, t[-1])
        shifted_time = telemetry.time + synchronization['shift']
        for channel in ('altitude', 'speed'):
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
            if len(unique) >= 3:
                samples.append({'phase': phase, 'channel': channel, 'time': unique,
                                'values': observations})
    if not samples:
        raise ValueError('No overlapping positive-acceleration first/second-stage telemetry'
                         + (' or booster return telemetry' if recovery != 'EXP' else '')
                         + ' with at least three distinct times.')
    return samples


def score_candidate(solution, telemetry, samples, primary_group='Stage 1 / booster',
                    altitude_scale=1000., speed_scale=25.):
    # Synchronization is deliberately performed inside EVERY scoring call.
    synchronization = synchronize_candidate(solution, telemetry, primary_group)
    errors, losses = {}, {}
    phases = {}
    for sample in samples:
        phase, channel = sample['phase'], sample['channel']
        if phase not in phases:
            phases[phase] = simulation_channels(solution, phase)
        time_sim, predicted = phases[phase]
        times = sample['time'] + synchronization['shift']
        interpolated = np.interp(times, time_sim, predicted[channel], left=np.nan, right=np.nan)
        valid = np.isfinite(interpolated)
        if valid.mean() < .9:
            raise ValueError(f'{phase} candidate covers less than 90% of the fixed telemetry window.')
        residual = interpolated[valid] - sample['values'][valid]
        scale = altitude_scale if channel == 'altitude' else speed_scale
        normalized = residual / scale
        robust = 2 * (np.hypot(1., normalized) - 1.)
        # Missing coverage has a large fixed penalty, never a reduced denominator.
        losses.setdefault(phase, []).append(float((robust.sum() + 10000 * (~valid).sum()) / len(times)))
        key = f'{phase}.{channel}'
        errors[key] = {'rmse': float(np.sqrt(np.mean(residual**2))),
                       'unit': 'm' if channel == 'altitude' else 'm/s',
                       'samples': int(valid.sum()), 'coverage': float(valid.mean())}
    return {'score': float(np.mean([np.mean(values) for values in losses.values()])), 'errors': errors, 'synchronization': synchronization}


class _StopSearch(Exception):
    pass


def fit_dispersion_to_telemetry(base_request, telemetry, *, parameters=DEFAULT_PARAMETERS,
                                bounds=None, primary_group='Stage 1 / booster', max_evaluations=80,
                                altitude_scale=1000., speed_scale=25., regularization=.01,
                                initial_solution=None, windows=None, progress_callback=None,
                                checkpoint_callback=None, cancel_callback=None,
                                solver=None, ipopt_options=None):
    """Return the best successful fit, its trajectory, baseline, and diagnostics.

    max_evaluations counts all solver calls, including optional payload bootstrap.
    A nonpositive payload is fixed from initial_solution, or from one preliminary
    nominal solve. All subsequent candidates use that SAME fixed payload.
    Bounds contain absolute factor/offset values. Progress and checkpoint hooks
    run after evaluations; cancellation is checked between solver calls.
    Non-EXP recovery includes available post-separation booster altitude/speed,
    without the positive-acceleration filter used for ascent. All phases share
    the first-/second-stage synchronization and fixed baseline sample windows.
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
    if not np.isfinite([altitude_scale, speed_scale, regularization]).all() or min(altitude_scale, speed_scale) <= 0 or regularization < 0:
        raise ValueError('Error scales must be positive; regularization must be nonnegative.')
    if primary_group not in ('Stage 1 / booster', 'Stage 2', 'Booster'):
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
            metrics = score_candidate(solution, telemetry, samples, primary_group, altitude_scale, speed_scale)
            penalty = float(regularization * np.mean((normalized - initial)**2))
            score = metrics['score'] + penalty
            candidate = dict(metrics, score=score, data_score=metrics['score'], request=candidate_request,
                             solution=solution, evaluation=calls)
            if baseline is None:
                baseline = candidate
            warm = solution
            record.update(score=score, status='OK', shift=metrics['synchronization']['shift'], errors=metrics['errors'])
            if best is None or score < best['score']:
                best = candidate
                if checkpoint_callback:
                    checkpoint_callback({'best': best, 'baseline': baseline, 'history': history + [record],
                                         'evaluations': calls, 'termination': 'Searching',
                                         'parameters': list(parameters), 'bounds': dict(zip(parameters, limits.tolist()))})
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
            'settings': {'altitude_scale': altitude_scale, 'speed_scale': speed_scale,
                         'regularization': regularization, 'primary_group': primary_group},
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
