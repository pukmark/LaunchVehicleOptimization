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


def default_steps(request):
    """One percent of physical quantities, including mass offsets centered on zero."""
    model = VLType()
    model.ApplyVehicleConfiguration(request['configuration'])
    model.ApplyScenarioDispersionToLV(DispesrionFactorsType(**request['dispersion']))
    masses = {'FirstStage_EmptyMass': model.EmptyFirstStageMass,
              'SecondStage_EmptyMass': model.EmptySecondStageMass,
              'FirstStage_PropMass': model.FirstStagePropellentMass,
              'SecondStage_PropMass': model.SecondStagePropellentMass}
    return {name: (.01 * request['dispersion'][name] if name in MULTIPLIERS else
                   .01 * masses[name] if name in masses else
                   100. if name == 'LaunchAltDelta' else .001)
            for name in asdict(DispesrionFactorsType())}


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
    for name, value in steps.items():
        if name not in request['dispersion']:
            raise ValueError(f'Unknown dispersion parameter: {name}')
        try:
            step = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name}: enter a positive step.') from None
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f'{name}: step must be positive and finite.')
        for sign in (-1, 1):
            candidate = deepcopy(request)
            candidate['dispersion'][name] += sign * step
            if candidate['dispersion'][name] == request['dispersion'][name]:
                raise ValueError(f'{name}: step is too small to change the parameter.')
            try:
                _validate_request(candidate)
            except ValueError as exc:
                raise ValueError(f'{name}: invalid {"minus" if sign < 0 else "plus"} step: {exc}') from None
        parsed[name] = step
    return parsed


def compute_sensitivities(base_request, steps, *, solve=None, initial_solution=None,
                          checkpoint=None, cancel_callback=None):
    """Hold all other settings fixed and estimate d(max payload)/d(parameter).

    Failed optimizations never contribute to a derivative. Each perturbed solve
    starts from the same newly optimized baseline to avoid sweep-order effects.
    """
    request = sensitivity_request(base_request)
    steps = validate_steps(request, steps)
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
    baseline, payload, error = evaluate(request, initial_solution)
    report['baseline'] = dict(payload=payload, error=error)
    if error:
        report['termination'] = 'Baseline failed; no sensitivities computed'
        publish()
        return report
    publish()
    for name, step in steps.items():
        if stopped():
            break
        row = dict(parameter=name, base=request['dispersion'][name], step=step,
                   minus_payload=None, plus_payload=None, derivative=None,
                   delta_payload=None, percent=None, status='Running', errors={})
        report['rows'].append(row)
        for side, sign in (('minus', -1), ('plus', 1)):
            if stopped():
                break
            candidate = deepcopy(request)
            candidate['dispersion'][name] += sign * step
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


def draw_sensitivities(figure, report):
    figure.clear()
    rows = [row for row in (report or {}).get('rows', []) if row.get('delta_payload') is not None]
    if not rows:
        figure.text(.5, .5, 'Run sensitivities to compare maximum payload changes.', ha='center', va='center')
        return
    rows.sort(key=lambda row: abs(row['delta_payload']))
    axis = figure.add_subplot(111)
    values = [row['delta_payload'] for row in rows]
    axis.barh([row['parameter'] for row in rows], values,
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

    compute_sensitivities(**settings, initial_solution=initial, checkpoint=checkpoint,
                          cancel_callback=lambda: (directory / 'CANCEL').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    worker(parser.parse_args().worker)
