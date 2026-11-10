"""Portable sensitivity result tables and run-time reproducibility snapshots."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import casadi
import numpy as np

from LaunchSites import parse_launch_site
from LV_Type_scaled import DispesrionFactorsType, VLType
from VehicleDefinitions import VEHICLE_PARAMETERS, vehicle_name


def _si_unit(display_unit):
    return {'t': 'kg', 'tf': 'N', 'km': 'm', 'kPa': 'Pa', 'kPa·rad': 'Pa·rad',
            'deg': 'rad', 'kW/m²': 'W/m²'}.get(display_unit, display_unit)


def _json_value(value):
    if isinstance(value, (np.ndarray, np.generic, casadi.DM)):
        return np.asarray(value).tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def warm_start_snapshot(solution):
    """Save the trajectory/control fields consumed by the optimizer, in SI."""
    if solution is None:
        return None
    keys = {'configuration', 'launch_site', 'solution_units', 'payload_mass', 'LaunchAz',
            'landing_thrust_angle', 'final_burn_coast_angle', 'final_burn_velocity_after'}
    selected = {key: value for key, value in solution.items()
                if key in keys or key.startswith(('x1', 'x2', 'x3', 'x4', 'u1', 'u2', 'u3', 'u4', 'dt'))}
    return json.loads(json.dumps(selected, default=_json_value))


def run_snapshot(request, initial_solution, *, custom_solver=False):
    model = VLType()
    model.ApplyVehicleConfiguration(request['configuration'])
    nominal = {name: float(getattr(model, name)) for name in VEHICLE_PARAMETERS}
    model.ApplyScenarioDispersionToLV(DispesrionFactorsType(**request['dispersion']))
    effective = {name: float(getattr(model, name)) for name in VEHICLE_PARAMETERS}
    library = os.environ.get('LV_IPOPT_HSL_LIBRARY', '/usr/local/lib/libcoinhsl.so')
    return dict(
        schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        vehicle_name=vehicle_name(request['configuration']), nominal_vehicle_parameters_si=nominal,
        effective_vehicle_parameters_si=effective,
        parameter_units={name: _si_unit(item[2])
                         for name, item in VEHICLE_PARAMETERS.items()},
        launch_site=parse_launch_site(request.get('launch_site')),
        initial_solution=warm_start_snapshot(initial_solution),
        cold_start_seed=0.5,
        method='Central difference; independent plus/minus solves use the optimized baseline warm start',
        solver='custom callback' if custom_solver else 'CasADi/IPOPT; defaults recorded by source hashes',
        solver_environment={key: os.environ.get(key) for key in
                            ('LV_IPOPT_LINEAR_SOLVER', 'LV_IPOPT_HSL_LIBRARY', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        linear_solver=os.environ.get('LV_IPOPT_LINEAR_SOLVER', 'ma97' if Path(library).is_file() else 'mumps'),
        versions=dict(python=platform.python_version(), numpy=np.__version__, casadi=casadi.__version__),
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(Path(__file__).parent.glob('*.py'))})


def export_sensitivity_csv(path, report):
    """One baseline/result per row; first row also holds the complete report JSON.

    Inputs come exclusively from the saved report. Legacy reports remain
    exportable, but missing historical vehicle values are never guessed.
    """
    if not report or not report.get('request'):
        raise ValueError('Run a sensitivity analysis before exporting results.')
    runs = report.get('runs') or [report]
    records = []
    for run in runs:
        request = run['request']
        snapshot = run.get('reproducibility') or {}
        orbit = request['target_orbit']
        site = snapshot.get('launch_site') or parse_launch_site(request.get('launch_site'))
        common = dict(
            recovery=request['recovery'], vehicle=snapshot.get('vehicle_name') or vehicle_name(request['configuration']),
            termination=run.get('termination'), baseline_payload_kg=(run.get('baseline') or {}).get('payload'),
            apogee_altitude_km=(orbit['apogee'] - 6378137.) / 1000.,
            perigee_altitude_km=(orbit['perigee'] - 6378137.) / 1000.,
            apogee_radius_m=orbit['apogee'], perigee_radius_m=orbit['perigee'],
            inclination_deg=float(np.rad2deg(orbit['i'])), inclination_rad=orbit['i'],
            direct_injection=orbit.get('direct_injection', False), launch_site=site['name'],
            launch_latitude_deg=site['latitude'], launch_longitude_deg=site['longitude'],
            launch_altitude_m=site['altitude'], payload_mass_predefined_kg=request['payload_mass_predefined'],
            reproducibility_note=('Run-time snapshot saved' if snapshot else
                                  'Legacy report: historical vehicle, solver and warm-start snapshot unavailable'))
        common.update({'dispersion.' + name: value for name, value in request['dispersion'].items()})
        for prefix, key in (('nominal_si.', 'nominal_vehicle_parameters_si'),
                            ('effective_si.', 'effective_vehicle_parameters_si')):
            common.update({prefix + name: value for name, value in snapshot.get(key, {}).items()})
        baseline = run.get('baseline') or {}
        records.append(dict(common, record_type='baseline', status=baseline.get('error') or run.get('termination'),
                            errors_json=json.dumps({'baseline': baseline['error']} if baseline.get('error') else {})))
        for row in run.get('rows', []):
            parameter = row['parameter']
            unit = ('kg' if parameter.endswith('Mass') else 'm' if parameter == 'LaunchAltDelta' else
                    'fraction')
            if parameter in VEHICLE_PARAMETERS:
                unit = _si_unit(VEHICLE_PARAMETERS[parameter][2])
            records.append(dict(common, record_type='sensitivity', parameter=parameter, parameter_unit=unit,
                                base=row.get('base'), step=row.get('step'),
                                minus_payload_kg=row.get('minus_payload'), plus_payload_kg=row.get('plus_payload'),
                                derivative_kg_per_parameter_unit=row.get('derivative'),
                                delta_payload_kg=row.get('delta_payload'), percent=row.get('percent'),
                                status=row.get('status'), errors_json=json.dumps(row.get('errors', {}), ensure_ascii=False)))
    # JSON preserves all nested runs, selected steps, failures, baseline guesses,
    # source fingerprints and original requests without flattening information away.
    records[0]['report_json'] = json.dumps(report, ensure_ascii=False, default=_json_value, allow_nan=False)
    leading = ['record_type', 'recovery', 'parameter', 'parameter_unit', 'base', 'step',
               'baseline_payload_kg', 'minus_payload_kg', 'plus_payload_kg',
               'derivative_kg_per_parameter_unit', 'delta_payload_kg', 'percent', 'status', 'errors_json']
    fields = leading + sorted(set().union(*(record.keys() for record in records)) - set(leading) - {'report_json'}) + ['report_json']
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return len(records)
