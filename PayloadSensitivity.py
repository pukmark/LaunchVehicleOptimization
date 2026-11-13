"""Local maximum-payload sensitivities, computed by independent trajectory solves."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pickle

import numpy as np

from DispersionFit import MULTIPLIERS, _validate_request, _solve_candidate, _atomic_json
from LV_Type_scaled import DispesrionFactorsType, VLType
from VehicleDefinitions import VEHICLE_PARAMETERS, validate_vehicle_definition, vehicle_name
from SensitivityCSV import run_snapshot, warm_start_snapshot


LAUNCH_PARAMETERS = ('FirstStage_MaxDynamicPressure', 'Booster_MaxDynamicPressure',
                     'FirstStage_StageSeparationMaxDynamicPressure',
                     'FairingSeparationAltitude', 'Payload_Max_acc')
LAUNCH_UNITS = dict(zip(LAUNCH_PARAMETERS, ('Pa', 'Pa', 'Pa', 'm', 'm/s²')))


def parameter_values(request):
    model = VLType()
    model.ApplyVehicleConfiguration(request['configuration'])
    return dict(asdict(DispesrionFactorsType(**request['dispersion'])),
                **{name: float(getattr(model, name)) for name in LAUNCH_PARAMETERS})


def validate_recoveries(request, recoveries=None):
    values = [request['recovery']] if recoveries is None else ([recoveries] if isinstance(recoveries, str) else list(recoveries))
    if not values or any(value not in ('EXP', 'ASDS', 'RTLS') for value in values):
        raise ValueError('Select at least one recovery strategy: EXP, ASDS, RTLS.')
    return list(dict.fromkeys(values))


def _parameter_request(request, steps):
    """Copy nominal vehicle limits without baking or reapplying dispersion."""
    request = deepcopy(request)
    if any(name in LAUNCH_PARAMETERS for name in steps):
        model = VLType()
        model.ApplyVehicleConfiguration(request['configuration'])
        request['configuration'] = validate_vehicle_definition(dict(
            name=vehicle_name(request['configuration']),
            parameters={name: float(getattr(model, name)) for name in VEHICLE_PARAMETERS}))
    return request


def _perturb(request, name, delta):
    candidate = deepcopy(request)
    if name in LAUNCH_PARAMETERS:
        candidate['configuration']['parameters'][name] += delta
        candidate['configuration'] = validate_vehicle_definition(candidate['configuration'])
    else:
        candidate['dispersion'][name] += delta
    _validate_request(candidate)
    return candidate


def default_steps(request):
    """One percent of physical quantities, including mass offsets centered on zero."""
    model = VLType()
    model.ApplyVehicleConfiguration(request['configuration'])
    model.ApplyScenarioDispersionToLV(DispesrionFactorsType(**request['dispersion']))
    masses = {'FirstStage_EmptyMass': model.EmptyFirstStageMass,
              'SecondStage_EmptyMass': model.EmptySecondStageMass,
              'FirstStage_PropMass': model.FirstStagePropellentMass,
              'SecondStage_PropMass': model.SecondStagePropellentMass}
    steps = {name: (.01 * request['dispersion'][name] if name in MULTIPLIERS else
                   .01 * masses[name] if name in masses else
                   100. if name == 'LaunchAltDelta' else .001)
            for name in asdict(DispesrionFactorsType())}
    steps.update({name: .01 * value if value > 0 else 100.
                  for name, value in parameter_values(request).items() if name in LAUNCH_PARAMETERS})
    return steps


def sensitivity_request(request):
    request = deepcopy(request)
    request['payload_mass_predefined'] = -1.
    request['dispersion'] = dict(asdict(DispesrionFactorsType()), **request['dispersion'])
    _validate_request(request)
    return request


def validate_steps(request, steps):
    request = sensitivity_request(request)
    if not steps:
        raise ValueError('Select at least one parameter.')
    parsed = {}
    values = parameter_values(request)
    parameter_request = _parameter_request(request, steps)
    for name, value in steps.items():
        if name not in values:
            raise ValueError(f'Unknown sensitivity parameter: {name}')
        try:
            step = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name}: enter a positive step.') from None
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f'{name}: step must be positive and finite.')
        for sign in (-1, 1):
            if values[name] + sign * step == values[name]:
                raise ValueError(f'{name}: step is too small to change the parameter.')
            try:
                _perturb(parameter_request, name, sign * step)
            except ValueError as exc:
                raise ValueError(f'{name}: invalid {"minus" if sign < 0 else "plus"} step: {exc}') from None
        parsed[name] = step
    return parsed


def _dual_names(request, steps):
    """Factors whose payload derivative is available from inequality multipliers."""
    from LV_Optimization_Type import _ParameterBuffer
    return [name for name in steps if name in _ParameterBuffer.SUPPORTED]


def expected_sensitivity_evaluations(request, steps):
    """Solves the sensitivity tab will run for this selection."""
    request = sensitivity_request(request)
    dual = _dual_names(request, steps)
    central = len(steps) - len(dual)
    return (1 if dual else 0) + ((1 + 2 * central) if central else 0)


def _multiplier_rows(request, steps, names, solution, payload, error, *, equalities=False):
    rows = []
    duals = (solution or {}).get('parameter_duals') or {}
    for name in names:
        row = dict(parameter=name, base=request['dispersion'][name], step=steps[name],
                   minus_payload=None, plus_payload=None, derivative=None,
                   delta_payload=None, percent=None, status='Failed', errors={})
        record = duals.get(name)
        if error or payload is None or record is None:
            row['errors']['multiplier'] = error or 'Solver did not return a multiplier'
        else:
            derivative = float(record['payload_sensitivity'])
            delta = derivative * steps[name]
            percent = 100 * delta / payload
            if np.isfinite([derivative, delta, percent]).all():
                row.update(derivative=derivative, delta_payload=delta, percent=percent,
                           status='Equality multiplier' if equalities else 'Multiplier')
                if equalities:
                    row['equality_dual'] = float(record['equality_dual'])
                else:
                    row['upper_dual'] = float(record['upper_dual'])
                    row['lower_dual'] = float(record['lower_dual'])
            else:
                row['errors']['multiplier'] = 'Nonfinite multiplier'
        rows.append(row)
    return rows


def compute_reported_sensitivities(base_request, steps, *, solve=None, initial_solution=None,
                                   checkpoint=None, cancel_callback=None, parameter_eps=1e-6,
                                   recoveries=None, parameter_equalities=False):
    """Sensitivities shown in the tab.

    Every selected dispersion factor is read from one buffered solve. The sampled
    button is the path that re-solves at plus and minus each step.
    """
    request = sensitivity_request(base_request)
    steps = validate_steps(request, steps)
    strategies = validate_recoveries(request, recoveries)
    if len(strategies) > 1:
        return _compute_recoveries(
            request, steps, strategies, solve=solve, initial_solution=initial_solution,
            checkpoint=checkpoint, cancel_callback=cancel_callback, parameter_eps=parameter_eps,
            parameter_equalities=parameter_equalities,
            run=compute_reported_sensitivities,
            total_per_strategy=expected_sensitivity_evaluations(request, steps))
    request['recovery'] = strategies[0]
    dual = _dual_names(request, steps)
    central = {name: steps[name] for name in steps if name not in dual}
    if not dual:
        return compute_sensitivities(request, central, solve=solve, initial_solution=initial_solution,
                                     checkpoint=checkpoint, cancel_callback=cancel_callback,
                                     recoveries=[strategies[0]])
    solve = solve or _solve_candidate
    report = dict(request=request, steps=steps, baseline=None, rows=[],
                  evaluations=0, total=expected_sensitivity_evaluations(request, steps),
                  termination='Running')

    def publish():
        if checkpoint:
            checkpoint(deepcopy(report))

    def stopped():
        return cancel_callback is not None and cancel_callback()

    if stopped():
        report['termination'] = 'Cancelled'
        publish()
        return report
    try:
        solve_options = dict(lifted_parameters=list(dual), parameter_eps=parameter_eps)
        if parameter_equalities:
            solve_options['parameter_equalities'] = True
        solution = solve(deepcopy(request), init_guess=deepcopy(initial_solution), **solve_options)
        if not solution.get('success'):
            solution, payload, error = None, None, str(solution.get('return_status', 'Solver did not converge'))
        else:
            payload = float(np.asarray(solution['payload_mass']).item())
            error = None if np.isfinite(payload) and payload > 0 else 'Solver returned an invalid payload'
            if error:
                payload = None
    except Exception as exc:
        solution, payload, error = None, None, str(exc)
    finally:
        report['evaluations'] += 1
    report['baseline'] = dict(payload=payload, error=error)
    multiplier_rows = _multiplier_rows(request, steps, dual, solution, payload, error,
                                      equalities=parameter_equalities)
    report['rows'] = list(multiplier_rows)
    publish()
    if central and not stopped():
        def central_checkpoint(central_report):
            by_name = {row['parameter']: row for row in multiplier_rows + central_report['rows']}
            report['rows'] = [by_name[name] for name in steps if name in by_name]
            report['evaluations'] = 1 + central_report['evaluations']
            report['termination'] = central_report['termination']
            publish()

        central_report = compute_sensitivities(
            request, central, solve=solve,
            initial_solution=solution if error is None else initial_solution,
            checkpoint=central_checkpoint, cancel_callback=cancel_callback)
        by_name = {row['parameter']: row for row in multiplier_rows + central_report['rows']}
        report['rows'] = [by_name[name] for name in steps if name in by_name]
        report['evaluations'] = 1 + central_report['evaluations']
        if report['baseline'].get('payload') is None:
            report['baseline'] = central_report['baseline']
    report['termination'] = ('Cancelled; partial results' if stopped() else
                             'Complete with failed parameters' if any(row['errors'] for row in report['rows']) else
                             'Complete')
    publish()
    return report


def compute_sensitivities(base_request, steps, *, solve=None, initial_solution=None,
                          checkpoint=None, cancel_callback=None, recoveries=None):
    """Hold all other settings fixed and estimate d(max payload)/d(parameter).

    Failed optimizations never contribute to a derivative. Each perturbed solve
    starts from the same newly optimized baseline to avoid sweep-order effects.
    """
    request = sensitivity_request(base_request)
    steps = validate_steps(request, steps)
    strategies = validate_recoveries(request, recoveries)
    if len(strategies) > 1:
        return _compute_recoveries(request, steps, strategies, solve=solve,
                                   initial_solution=initial_solution, checkpoint=checkpoint,
                                   cancel_callback=cancel_callback)
    request['recovery'] = strategies[0]
    solver_request = _parameter_request(request, steps)
    custom_solver = solve is not None
    solve = solve or _solve_candidate
    report = dict(request=request, steps=steps, baseline=None, rows=[],
                  evaluations=0, total=1 + 2 * len(steps), termination='Running')

    def publish():
        if checkpoint:
            checkpoint(deepcopy(report))

    def stopped():
        return cancel_callback is not None and cancel_callback()

    def evaluate(candidate, guess):
        try:
            solution = solve(deepcopy(candidate), init_guess=deepcopy(guess))
            if not solution.get('success'):
                return None, None, str(solution.get('return_status', 'Solver did not converge'))
            payload = float(np.asarray(solution['payload_mass']).item())
            if not np.isfinite(payload) or payload <= 0:
                return None, None, 'Solver returned an invalid payload'
            return solution, payload, None
        except Exception as exc:
            return None, None, str(exc)
        finally:
            report['evaluations'] += 1

    if stopped():
        report['termination'] = 'Cancelled'
        publish()
        return report
    guess = initial_solution if request['recovery'] == base_request['recovery'] else None
    if (guess is not None and isinstance(solver_request['configuration'], dict)
            and guess.get('configuration', request['configuration']) == request['configuration']):
        guess = deepcopy(guess)
        guess['configuration'] = deepcopy(solver_request['configuration'])
    report['reproducibility'] = run_snapshot(solver_request, guess, custom_solver=custom_solver)
    report['solver_request'] = deepcopy(solver_request)
    baseline, payload, error = evaluate(solver_request, guess)
    report['baseline'] = dict(payload=payload, error=error)
    if baseline is not None:
        report['reproducibility']['baseline_solution'] = warm_start_snapshot(baseline)
    if error:
        report['termination'] = 'Baseline failed; no sensitivities computed'
        publish()
        return report
    publish()
    for name, step in steps.items():
        if stopped():
            break
        row = dict(parameter=name, base=parameter_values(request)[name], step=step,
                   minus_payload=None, plus_payload=None, derivative=None,
                   delta_payload=None, percent=None, status='Running', errors={})
        report['rows'].append(row)
        for side, sign in (('minus', -1), ('plus', 1)):
            if stopped():
                break
            candidate = _perturb(solver_request, name, sign * step)
            _, mass, error = evaluate(candidate, baseline)
            row[side + '_payload'] = mass
            if error:
                row['errors'][side] = error
            publish()
        if row['minus_payload'] is not None and row['plus_payload'] is not None:
            delta = (row['plus_payload'] - row['minus_payload']) / 2
            derivative = delta / step
            percent = 100 * (delta / payload)
            if np.isfinite([derivative, delta, percent]).all():
                row.update(derivative=derivative, delta_payload=delta,
                           percent=percent, status='Central difference')
            else:
                row['errors']['difference'] = 'Nonfinite sensitivity; choose a larger step'
                row['status'] = 'Failed'
        else:
            row['status'] = 'Failed' if row['errors'] else 'Cancelled'
        publish()
    report['termination'] = ('Cancelled; partial results' if stopped() else
                             'Complete with failed parameters' if any(row['errors'] for row in report['rows']) else
                             'Complete')
    publish()
    return report


def _compute_recoveries(request, steps, strategies, *, run=None, total_per_strategy=None, **options):
    run = run or compute_sensitivities
    if total_per_strategy is None:
        total_per_strategy = 1 + 2 * len(steps)
    checkpoint = options.pop('checkpoint')
    cancel = options.get('cancel_callback')
    initial = options.pop('initial_solution')
    report = dict(request=request, steps=steps, recoveries=strategies, runs=[], rows=[],
                  baseline=None, evaluations=0, total=len(strategies) * total_per_strategy,
                  termination='Running')
    for strategy in strategies:
        if cancel and cancel():
            break
        completed = list(report['runs'])

        def publish(current):
            report['runs'] = completed + [current]
            report['rows'] = [dict(row, recovery=run['request']['recovery'])
                              for run in report['runs'] for row in run['rows']]
            report['evaluations'] = sum(run['evaluations'] for run in report['runs'])
            if checkpoint:
                checkpoint(deepcopy(report))

        run(request, steps, recoveries=[strategy], checkpoint=publish,
            initial_solution=initial if strategy == request['recovery'] else None, **options)
    report['termination'] = ('Cancelled; partial results' if cancel and cancel() else
                             'Complete with failed strategies or parameters'
                             if any(run['termination'] != 'Complete' for run in report['runs']) else 'Complete')
    if checkpoint:
        checkpoint(deepcopy(report))
    return report


def draw_sensitivities(figure, report):
    figure.clear()
    rows = [row for row in (report or {}).get('rows', []) if row.get('delta_payload') is not None]
    if not rows:
        figure.text(.5, .5, 'Run sensitivities to compare maximum payload changes.', ha='center', va='center')
        return
    rows.sort(key=lambda row: abs(row['delta_payload']))
    axis = figure.add_subplot(111)
    values = [row['delta_payload'] for row in rows]
    axis.barh([f"{row['parameter']} · {row['recovery']}" if 'recovery' in row else row['parameter'] for row in rows], values,
              color=['#009E73' if value >= 0 else '#D55E00' for value in values])
    axis.axvline(0, color='#777777', linewidth=.8)
    axis.set_xlabel('Estimated maximum-payload change for +one configured step [kg]')
    axis.set_title('Local payload sensitivity · other parameters held fixed')
    axis.grid(axis='x', alpha=.25)


def worker(directory):
    directory = Path(directory)
    settings = json.loads((directory / 'sensitivity_inputs.json').read_text())
    initial = None
    if (directory / 'initial_solution.pickle').exists():
        with (directory / 'initial_solution.pickle').open('rb') as stream:
            initial = pickle.load(stream)

    def checkpoint(report):
        _atomic_json(directory / 'sensitivity_report.json', report)
        print(f"Sensitivity solve {report['evaluations']}/{report['total']} · {report['termination']}", flush=True)

    method = settings.pop('method', 'multipliers')
    if method not in {'multipliers', 'equalities', 'samples'}:
        raise ValueError(f'Unknown sensitivity method: {method}')
    run = compute_sensitivities if method == 'samples' else compute_reported_sensitivities
    if method == 'equalities':
        settings['parameter_equalities'] = True
    run(**settings, initial_solution=initial, checkpoint=checkpoint,
        cancel_callback=lambda: (directory / 'CANCEL').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    worker(parser.parse_args().worker)
