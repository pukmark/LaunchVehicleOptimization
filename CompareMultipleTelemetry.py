#!/usr/bin/env python3
"""Optimize SXM-11, CSG-3 and MRV–MEP and compare their ascent telemetry.

Run with the repository's Python environment::

    .venv/bin/python CompareMultipleTelemetry.py
    .venv/bin/python CompareMultipleTelemetry.py --mrv-payload 7000 --no-show

The default vehicle is Falcon 9 True (configuration 5), with nominal dispersion.
Mission reference values below are supplied by the assessment mission table.
MRV payload is unknown: without --mrv-payload, maximize payload and label the
result as a model estimate, not a measured payload. Launch coordinates use the
repository's regional Cape/Kennedy and Vandenberg presets, not surveyed pads.

CSG-3 uses direct orbital injection. The two GTO missions use the existing
parking-orbit plus analytical propellant-reserve model. Their later coast and
restart are NOT propagated or drawn. Phases 2 and 3 split one continuous burn
at fairing jettison; they are not separate engine burns. Broadcast scalar speed
is treated as ECEF speed, consistently with TelemetryCSV and the GUI.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from IntegratedResults import integrated_telemetry_curves
from LaunchSites import launch_site_inputs
from LV_Optimization_Type import LV_Optimization
from LV_Type_scaled import DispesrionFactorsType
from PaperResults import paper_histories
from PlotLegends import set_vehicle_legend
from TelemetryCSV import TelemetryData, load_telemetry_csv
from VehicleDefinitions import BUILTIN_VEHICLES, vehicle_name

ROOT = Path(__file__).resolve().parent
R0 = 6_378_137.0
DEFAULT_CONFIGURATION = BUILTIN_VEHICLES['Falcon 9 True']


@dataclass(frozen=True)
class Mission:
    name: str
    telemetry_file: str
    launch_date: str
    launch_site: str
    launch_pad: str
    payload_kg: float | None
    apogee_km: float
    perigee_km: float
    inclination_deg: float
    recovery: str
    orbit_reference: str

    def target_orbit(self):
        return dict(Name=self.name, apogee=R0 + self.apogee_km * 1000.,
                    perigee=R0 + self.perigee_km * 1000.,
                    i=np.deg2rad(self.inclination_deg),
                    direct_injection=False)


MISSIONS = (
    Mission('SXM-11', 'falcon9_telemetry_SXM11.csv', '2026-06-29',
            'Kennedy Space Center', 'SLC-40', 6950., 19463., 204., 26.48,
            'ASDS', 'Prelaunch injection-orbit estimate'),
    Mission('SPHEREx', 'falcon9_SPHEREx_SSO.csv', '2025-03-11',
            'Vandenberg', 'SLC-4E', 758., 660., 646., 97.96,
            'RTLS', 'Deployment-orbit reference'),
    Mission('MRV–MEP', 'falcon9_telemetry_MRV.csv', '2026-07-21',
            'Kennedy Space Center', 'SLC-40', 4200, 44765., 309., 25.63,
            'EXP', 'Early tracking estimate, approximately two days after launch'),
)


@dataclass
class Comparison:
    mission: Mission
    telemetry: TelemetryData
    solution: dict | None = None
    error: str = ''
    payload_is_estimate: bool = False
    telemetry_shift: float = 0.


def optimize_mission(mission, *, configuration=DEFAULT_CONFIGURATION, mrv_payload=None,
                     max_iter=800, max_cpu_time=120., print_ipopt=True):
    """Return only a successful solution; never plot a failed solver iterate."""
    payload = mission.payload_kg if mission.payload_kg is not None else mrv_payload
    optimizer = LV_Optimization(DispesrionFactorsType(),
                                LV_Configuration=configuration,
                                launch_site=launch_site_inputs(mission.launch_site))
    # Scalar seeds affect the guessed payload only in maximize-payload mode.
    seeds = (0.5, .35, .65) if payload is None else (.5,)
    statuses = []
    for seed in seeds:
        solution = optimizer.SolveOptimiztion(
            RecoveryStrategy=mission.recovery,
            payload_mass_predefined=-1. if payload is None else payload,
            Target_Orbit=mission.target_orbit(), Plot_interm=False,
            print_ipopt=print_ipopt, init_guess=seed,
            ipopt_options=dict(max_iter=max_iter, max_cpu_time=max_cpu_time),
            output_dir=None, initial_guess_diagnostics=False)
        if solution.get('success', False):
            return solution
        statuses.append(str(solution.get('return_status', 'Unknown solver failure')))
    raise RuntimeError('; '.join(statuses))


def run_comparisons(*, telemetry_dir=ROOT / 'falcon9_telemetry', mrv_payload=None,
                    time_shifts=(0., 0., 0.), **solver_options):
    """Load all inputs before running the independent mission optimizations."""
    comparisons = [Comparison(mission, load_telemetry_csv(Path(telemetry_dir) / mission.telemetry_file),
                              payload_is_estimate=mission.payload_kg is None and mrv_payload is None,
                              telemetry_shift=float(shift))
                   for mission, shift in zip(MISSIONS, time_shifts, strict=True)]
    for comparison in comparisons:
        mission = comparison.mission
        print(f'\n{mission.name}: {mission.recovery}, {mission.launch_site} / {mission.launch_pad}, '
              f'{mission.apogee_km:g} × {mission.perigee_km:g} km, '
              f'{mission.inclination_deg:g} deg', flush=True)
        print(f'  {mission.orbit_reference}.', flush=True)
        try:
            comparison.solution = optimize_mission(mission, mrv_payload=mrv_payload, **solver_options)
            mass = comparison.solution['payload_mass']
            qualifier = 'optimized model estimate; unverified' if comparison.payload_is_estimate else 'fixed'
            print(f'  Converged; payload {mass:,.1f} kg ({qualifier}).', flush=True)
        except (RuntimeError, ValueError) as exc:
            comparison.error = str(exc)
            print(f'  Optimization failed: {exc}', flush=True)
    return comparisons


def plot_comparisons(comparisons):
    """Create one 2×3 figure; preserve telemetry gaps and mission timestamps."""
    if len(comparisons) != 3:
        raise ValueError('The comparison requires exactly three missions.')
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col', layout='constrained')
    for column, comparison in enumerate(comparisons):
        mission, solution = comparison.mission, comparison.solution
        groups = set()
        if solution is not None:
            for phase, time, altitude, speed in paper_histories(solution):
                booster = phase == 2
                groups.add('Booster' if booster else 'Main vehicle')
                for row, values in ((0, speed), (1, altitude)):
                    axes[row, column].plot(time, values, color='#0072B2',
                                           linestyle='--' if booster else '-', linewidth=1.4)
        curves = integrated_telemetry_curves(
            comparison.telemetry, 'Phase columns', [solution] if solution is not None else [],
            comparison.telemetry_shift)
        for index, time, values, label, _ in curves:
            if index not in (0, 1):
                continue
            groups.add('Booster' if 'Booster return' in label else 'Main vehicle')
            axes[1 - index, column].plot(time, values, linestyle='none', marker='.',
                                         markersize=2.5, color='#D55E00', alpha=.7)
        title = f'{mission.name} · {mission.recovery}'
        if comparison.payload_is_estimate:
            estimate = f"{solution['payload_mass']:,.0f} kg" if solution is not None else 'unknown'
            title += f'\nPayload: {estimate} (model estimate, unverified)'
        else:
            mass = solution['payload_mass'] if solution is not None else mission.payload_kg
            title += f'\nPayload: {mass:,.0f} kg' if mass is not None else '\nPayload: unverified'
        axes[0, column].set_title(title, fontsize=11)
        azimuth = float(np.asarray((solution or {}).get('LaunchAz', np.nan)).item())
        delta_v = float(np.asarray((solution or {}).get('final_burn_delta_v', np.nan)).item())
        azimuth_text = f'{np.rad2deg(azimuth):.2f}°' if np.isfinite(azimuth) else 'unavailable'
        if (solution or {}).get('direct_injection', mission.target_orbit().get('direct_injection', False)):
            delta_v_text = 'N/A (single burn)'
        else:
            delta_v_text = f'{delta_v:,.1f} m/s' if np.isfinite(delta_v) else 'unavailable'
        axes[0, column].text(
            .03, .97, f'Optimized launch azimuth: {azimuth_text}\nSecond SES Δv: {delta_v_text}',
            transform=axes[0, column].transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        if comparison.error:
            axes[0, column].text(.97, .04, 'Optimization failed\nTelemetry only',
                                 transform=axes[0, column].transAxes, ha='right', va='bottom',
                                 color='firebrick', bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        for row, quantity in ((0, 'Velocity, ECEF [m/s]'), (1, 'Altitude [km]')):
            axis = axes[row, column]
            axis.set_ylabel(quantity)
            axis.grid(True, linewidth=.4, alpha=.3)
            axis.set_xlim(left=0.)
            legend_options = dict(loc='upper left', bbox_to_anchor=(0., .80)) if row == 0 else dict(loc='best')
            set_vehicle_legend(axis, groups, fontsize=8, frameon=False, **legend_options)
        axes[1, column].set_xlabel('Time since launch [s]')
    vehicles = sorted({vehicle_name(case.solution.get('configuration', DEFAULT_CONFIGURATION))
                       for case in comparisons if case.solution is not None})
    vehicle_label = ' / '.join(vehicles) or vehicle_name(DEFAULT_CONFIGURATION)
    figure.suptitle(f'{vehicle_label} mission comparisons — blue lines: optimization; orange dots: telemetry',
                   fontsize=12)
    figure.supxlabel('GTO curves end at first cutoff; later coast/restart is not propagated. '
                       'Launch coordinates use regional presets.', fontsize=9)
    return figure, axes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--telemetry-dir', type=Path, default=ROOT / 'falcon9_telemetry')
    parser.add_argument('--configuration', type=int, default=DEFAULT_CONFIGURATION, help='Vehicle configuration; default: 5 (Falcon 9 True).')
    parser.add_argument('--mrv-payload', type=float, help='Fixed combined MRV + three MEP payload [kg]. Default: optimize an estimate.')
    parser.add_argument('--time-shifts', type=float, nargs=3, default=(0., 0., 0.),
                        metavar=('SXM11', 'CSG3', 'MRV'), help='Seconds added to telemetry mission times; default: no shifts.')
    parser.add_argument('--max-iter', type=int, default=800)
    parser.add_argument('--max-cpu-time', type=float, default=120., help='IPOPT CPU seconds per attempt.')
    parser.add_argument('--print-ipopt', default=False, action='store_true')
    parser.add_argument('--output', type=Path, default=Path('CompareMultipleTelemetry.png'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args(argv)
    for name, value in (('MRV payload', args.mrv_payload), ('iteration limit', args.max_iter),
                        ('CPU time limit', args.max_cpu_time)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f'{name} must be positive and finite.')
    if not np.isfinite(args.time_shifts).all():
        parser.error('Time shifts must be finite.')
    print(f'Comparison vehicle: {vehicle_name(args.configuration)} (configuration {args.configuration}).', flush=True)
    print('GTO missions use parking-orbit ascent with a later-burn propellant reserve; '
          'only the propagated ascent is plotted. CSG-3 uses direct injection.', flush=True)
    comparisons = run_comparisons(telemetry_dir=args.telemetry_dir, mrv_payload=args.mrv_payload,
                                  time_shifts=args.time_shifts, configuration=args.configuration,
                                  max_iter=args.max_iter, max_cpu_time=args.max_cpu_time,
                                  print_ipopt=args.print_ipopt)
    figure, _ = plot_comparisons(comparisons)
    figure.savefig(args.output, dpi=200)
    print(f'Comparison saved to {args.output.resolve()}', flush=True)
    if not args.no_show:
        plt.show()
    return int(any(comparison.error for comparison in comparisons))


if __name__ == '__main__':
    raise SystemExit(main())
