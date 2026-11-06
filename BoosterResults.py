"""Altitude-oriented booster return plots for simulation and telemetry."""
import numpy as np

from LV_Type_scaled import VLType
from TelemetryCSV import aerodynamic_loads


def draw_booster_altitude_results(figure, cases, *, telemetry=None, primary_group='Phase columns',
                                  telemetry_color='#CC79A7', telemetry_shift=0.):
    """Each case is (SI solution, color, label); altitude is always vertical."""
    figure.clear()
    grid = figure.add_gridspec(2, 6)
    axes = [figure.add_subplot(grid[0, :2])]
    axes += [figure.add_subplot(grid[0, 2:4], sharey=axes[0]),
             figure.add_subplot(grid[0, 4:], sharey=axes[0]),
             figure.add_subplot(grid[1, :3], sharey=axes[0]),
             figure.add_subplot(grid[1, 3:], sharey=axes[0])]
    for solution, color, label in cases:
        if 'x4' not in solution:
            continue
        state = np.asarray(solution['x4'])
        altitude = np.asarray(solution['alt4_sol']).ravel() / 1000.
        speed = np.linalg.norm(state[:, 2:4], axis=1)
        model = VLType()
        model.ApplyVehicleConfiguration(solution.get('configuration', 1))
        loads = aerodynamic_loads(altitude * 1000., speed,
                                  density_factor=solution.get('dispersion', {}).get('AtmosphereDensity', 1.),
                                  heat_coefficient=model.Booster_k_empirical)
        values = (speed, state[:, 4] / 1000., np.asarray(solution['Qdyn4_sol']).ravel() / 1000.,
                  loads['heat_flux'], np.asarray(solution['acc4_sol']).ravel())
        for index, (axis, value) in enumerate(zip(axes, values)):
            curve_label = f'{label} · specific' if index == 4 else label
            axis.plot(value, altitude, '--', color=color, linewidth=1.6, label=curve_label)

    overlaid = False
    if telemetry is not None:
        channels = telemetry.channels('booster', primary_group)
        explicit = telemetry.channels('booster', 'Phase columns')
        if primary_group == 'Stage 1 / booster' and telemetry.groups['primary'] and cases:
            # Route only the shared primary feed by simulation time. Explicit
            # booster measurements retain their complete extent.
            time = np.asarray(telemetry.time) + telemetry_shift
            mask = np.zeros(len(time), dtype=bool)
            for solution, _, _ in cases:
                if 't4_vec' in solution:
                    mask |= (time >= np.min(solution['t4_vec'])) & (time <= np.max(solution['t4_vec']))
            channels = {key: np.where(mask, value, explicit.get(key, np.full(len(time), np.nan)))
                        for key, value in channels.items()}
        if 'altitude' in channels:
            altitude = np.asarray(channels['altitude'])
            for index, key in ((0, 'speed'), (1, 'mass'), (2, 'pressure'), (2, 'pressure_derived'),
                               (3, 'heat_flux'), (3, 'heat_flux_derived'),
                               (4, 'acceleration'), (4, 'acceleration_derived')):
                if key not in channels:
                    continue
                values = np.asarray(channels[key])
                if not (np.isfinite(values) & np.isfinite(altitude)).any():
                    continue
                label = 'Telemetry'
                if key.endswith('_derived'):
                    label += ' · dv/dt (ECEF)' if key == 'acceleration_derived' else ' · derived'
                elif key in ('pressure', 'heat_flux', 'acceleration'):
                    label += ' · CSV'
                axes[index].plot(values, altitude, ':', marker='.', markersize=3,
                                 color=telemetry_color, linewidth=1.2, label=label)
                overlaid = True

    for axis, title, xlabel in zip(axes,
            ('Velocity (Earth-fixed)', 'Mass', 'Dynamic pressure', 'Heat flux', 'Acceleration: specific / dv/dt'),
            ('Speed [m/s]', 'Mass [t]', 'Dynamic pressure [kPa]', 'Heat flux [kW/m²]', 'Acceleration [m/s²]')):
        axis.set(title=title, xlabel=xlabel, ylabel='Altitude [km]')
        axis.grid(True, alpha=.25)
        axis.tick_params(labelsize=8)
        if axis.lines:
            axis.legend(fontsize=7, loc='best')
        else:
            axis.text(.5, .5, 'No matching data', transform=axis.transAxes, ha='center', color='#666666')
    figure.suptitle('Booster return versus altitude', fontsize=12)
    return overlaid
