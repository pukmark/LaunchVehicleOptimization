"""Compare parameter-inequality duals with central-difference payload sensitivities.

Each dispersion factor is a decision variable boxed by two inequalities,

    nominal - eps <= factor <= nominal + eps,

and is not fixed by an equality. The IPOPT dual of the upper inequality minus
the dual of the lower inequality estimates the derivative of maximized
performance. One solve returns every requested factor. Central differences
re-solve the same transcription at plus and minus one step.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import time

import numpy as np

from LaunchSites import parse_launch_site
from LV_Optimization_Type import _ParameterBuffer
from LV_Type_scaled import DispesrionFactorsType, VLType


def comparison_request():
    """Expendable maximum-payload case used by the sensitivity check."""
    radius = 6378137.0
    model = VLType()
    model.ApplyVehicleConfiguration(1)
    steps = {
        'SecondStageIsp': 0.01, 'FirstStageIsp': 0.01,
        'FirstStageThrust': 0.01, 'SecondStageThrust': 0.01,
        'FirstStageCx0': 0.01, 'BoosterStageCx0': 0.01, 'AtmosphereDensity': 0.01,
        'SecondStage_EmptyMass': 0.01 * model.EmptySecondStageMass,
        'FirstStage_EmptyMass': 0.01 * model.EmptyFirstStageMass,
        'SecondStage_PropMass': 0.01 * model.SecondStagePropellentMass,
        'FirstStage_PropMass': 0.01 * model.FirstStagePropellentMass,
        'LaunchAltDelta': 100., 'StagePartitionDelta': 0.001,
    }
    return {
        'launch_site': parse_launch_site(None),
        'dispersion': asdict(DispesrionFactorsType()),
        'configuration': 1,
        'recovery': 'EXP',
        'target_orbit': {'Name': 'LEO', 'apogee': radius + 300e3,
                         'perigee': radius + 200e3, 'i': float(np.deg2rad(28.6))},
        'payload_mass_predefined': -1.0,
    }, steps


def _payload(solution):
    return float(np.asarray(solution['payload_mass']).reshape(-1)[0])


def solve_case(request, horizon, init_guess=None, lifted=None, eps=1e-6, label='solve'):
    from LV_Optimization_Type import LV_Optimization
    started = time.time()
    dispersion = DispesrionFactorsType(**request['dispersion'])
    model = LV_Optimization(dispersion, request['configuration'],
                            launch_site=parse_launch_site(request.get('launch_site')))
    model.booster.N = horizon[0]
    model.eci.N = [horizon[1], horizon[2]]
    solution = model.SolveOptimiztion(
        RecoveryStrategy=request['recovery'], Target_Orbit=request['target_orbit'],
        payload_mass_predefined=request['payload_mass_predefined'], init_guess=init_guess,
        Plot_interm=False, print_ipopt=False, output_dir=None,
        initial_guess_diagnostics=False, lifted_parameters=lifted, parameter_eps=eps)
    elapsed = time.time() - started
    payload = _payload(solution) if solution.get('success') else None
    print(f'{label}: success={solution.get("success")} payload={payload} '
          f'objective={solution.get("objective")} status={solution.get("return_status")} '
          f'[{elapsed:.1f}s]', flush=True)
    solution['elapsed_seconds'] = elapsed
    return solution


def central_difference(request, name, step, horizon, baseline):
    payloads, objectives = {}, {}
    for side, sign in (('minus', -1.0), ('plus', 1.0)):
        candidate = deepcopy(request)
        candidate['dispersion'][name] = request['dispersion'][name] + sign * step
        solution = solve_case(candidate, horizon, init_guess=baseline, label=f'{name} {side}')
        if not solution.get('success') or solution.get('objective') is None:
            return None
        payloads[side] = _payload(solution)
        objectives[side] = float(solution['objective'])
    payload_derivative = (payloads['plus'] - payloads['minus']) / (2 * step)
    objective_derivative = (objectives['plus'] - objectives['minus']) / (2 * step)
    return {'step': step, 'minus_payload': payloads['minus'], 'plus_payload': payloads['plus'],
            'payload_derivative': payload_derivative,
            'objective_derivative': objective_derivative,
            'performance_derivative': -objective_derivative}


def compare(request, names, steps, horizon, eps):
    baseline = solve_case(request, horizon, label='baseline')
    if not baseline.get('success'):
        raise RuntimeError('Baseline optimization failed.')
    dual = solve_case(request, horizon, init_guess=baseline, lifted=names, eps=eps, label='dual buffer')
    if not dual.get('success') or 'parameter_duals' not in dual:
        raise RuntimeError('Buffered-parameter optimization failed.')
    rows = []
    for name in names:
        difference = central_difference(request, name, steps[name], horizon, baseline)
        record = dual['parameter_duals'][name]
        row = {'parameter': name, 'dual': record, 'central': difference}
        if difference is not None:
            row['payload_ratio'] = _ratio(record['payload_sensitivity'], difference['payload_derivative'])
            row['performance_ratio'] = _ratio(record['dual_sensitivity'], difference['performance_derivative'])
        rows.append(row)
        _print_row(row)
    return {'eps': eps, 'horizon': list(horizon),
            'baseline_payload': _payload(baseline), 'baseline_objective': baseline.get('objective'),
            'rows': rows}


def _ratio(dual, reference):
    if reference is None or not np.isfinite(reference) or abs(reference) < 1e-8:
        return None
    return float(dual / reference)


def _print_row(row):
    dual = row['dual']
    central = row['central']
    if central is None:
        print(f"{row['parameter']}: central difference failed", flush=True)
        return
    print(f"{row['parameter']}: dual payload {dual['payload_sensitivity']:.6g}, "
          f"central payload {central['payload_derivative']:.6g}, "
          f"ratio {row['payload_ratio']}, "
          f"dual performance {dual['dual_sensitivity']:.6g}, "
          f"central -dJ {central['performance_derivative']:.6g}, "
          f"ratio {row['performance_ratio']}, "
          f"pinned at {dual['value']:.8g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--ascent', type=int, default=60)
    parser.add_argument('--coast', type=int, default=10)
    parser.add_argument('--burn', type=int, default=60)
    parser.add_argument('--parameters', nargs='+', default=sorted(_ParameterBuffer.SUPPORTED))
    parser.add_argument('--output', default='dual_sensitivity_comparison.json')
    args = parser.parse_args()
    request, steps = comparison_request()
    report = compare(request, args.parameters, steps, (args.ascent, args.coast, args.burn), args.eps)
    with open(args.output, 'w') as stream:
        json.dump(report, stream, indent=2)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
