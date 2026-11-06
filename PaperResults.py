"""Publication figures comparing displayed trajectories with flight telemetry."""
import numpy as np
from matplotlib.lines import Line2D

from IntegratedResults import integrated_telemetry_curves, second_stage_coast
from TelemetryFrames import eci_to_ecef_velocity
from TelemetryCSV import numerical_acceleration


def paper_histories(solution):
    """Return (phase, mission time, altitude in km, ECEF speed in m/s) segments."""
    histories = []

    def plot(phase, time, altitude, speed):
        histories.append((phase, np.ravel(time), np.ravel(altitude), np.ravel(speed)))

    if 'x1' in solution:
        plot(0, solution['t1_vec'], np.ravel(solution['alt1_sol']) / 1000.,
             np.linalg.norm(np.asarray(solution['x1'])[2:4], axis=0))
    if all(key in solution for key in ('x1', 't1_vec', 't2_vec')):
        coast = second_stage_coast(solution)
        if coast is not None:
            time, state, altitude = coast
            plot(1, time, altitude / 1000.,
                 np.linalg.norm(eci_to_ecef_velocity(state[:3], state[3:6], time), axis=0))
    for number in (2, 3):
        if f'x{number}' not in solution:
            continue
        time, state = np.ravel(solution[f't{number}_vec']), np.asarray(solution[f'x{number}'])
        plot(1, time, np.ravel(solution[f'alt{number}_sol']) / 1000.,
             np.linalg.norm(eci_to_ecef_velocity(state[:3], state[3:6], time), axis=0))
    if 'x4' in solution:
        plot(2, solution['t4_vec'], np.ravel(solution['alt4_sol']) / 1000.,
             np.linalg.norm(np.asarray(solution['x4'])[:, 2:4], axis=1))
    return histories


def draw_paper_results(figure, cases, *, telemetry=None, primary_group='Phase columns',
                       telemetry_color='#CC79A7', telemetry_shift=0.):
    """Plot altitude, ECEF speed, and signed speed derivative for every phase.

    Differentiate each segment independently, preserving telemetry gaps and
    avoiding differences across burn/coast boundaries. Acceleration is d|v_ECEF|/dt,
    including gravity, rather than the supplied proper-acceleration channel.
    """
    figure.clear()
    figure.set_constrained_layout(True)
    grid = figure.add_gridspec(4, 1, height_ratios=(.12, 1., 1., 1.))
    axes = [figure.add_subplot(grid[1, 0])]
    axes.append(figure.add_subplot(grid[2, 0], sharex=axes[0]))
    axes.append(figure.add_subplot(grid[3, 0], sharex=axes[0]))
    curves = list(integrated_telemetry_curves(
        telemetry, primary_group, [s for s, _, _ in cases], telemetry_shift)) if telemetry is not None else []
    used_cases = []
    for solution, color, label in cases:
        histories = paper_histories(solution)
        for _, time, altitude, speed in histories:
            acceleration = numerical_acceleration(time, speed)
            for axis, values in zip(axes, (altitude, speed, acceleration)):
                axis.plot(time, values, color=color, linewidth=1.3)
        if histories:
            used_cases.append((color, label))

    has_telemetry = False
    for index, time, values, label, _ in curves:
        if index > 1:
            continue
        axes[index].plot(time, values, color=telemetry_color, linestyle='none',
                              marker='.', markersize=2.5, alpha=.8, zorder=3)
        if index == 1:
            axes[2].plot(time, numerical_acceleration(time, values),
                         color=telemetry_color, linestyle='none', marker='.',
                         markersize=2.5, alpha=.8, zorder=3)
        has_telemetry = True

    for axis, quantity in zip(axes, ('Altitude [km]', 'Velocity, ECEF [m/s]',
                                   'ECEF speed derivative [m/s²]')):
        axis.set_xlabel('Time since launch [s]', fontsize=9)
        axis.set_ylabel(quantity, fontsize=9)
        axis.tick_params(labelsize=8, direction='in')
        axis.grid(True, linewidth=.4, alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.margins(x=.02)
        if not axis.lines:
            axis.text(.5, .5, 'No data', transform=axis.transAxes,
                      ha='center', va='center', fontsize=9, color='.5')
    handles = [Line2D([], [], color=color, linewidth=1.3, label=label) for color, label in used_cases]
    if has_telemetry:
        handles.append(Line2D([], [], color=telemetry_color, linestyle='none', marker='.',
                              markersize=4, label='Telemetry'))
    if handles:
        figure.legend(handles=handles, loc='upper center', ncol=min(4, len(handles)),
                      frameon=False, fontsize=9)
    return has_telemetry


def paper_comparison_costs(cases, *, telemetry=None, primary_group='Phase columns',
                           telemetry_shift=0., altitude_scale=1000., speed_scale=25.):
    """Score plotted altitude/speed overlaps, balancing phases and channels.

    Use the displayed time shift, all observed samples, and the same reference
    frame and segments as the figure. Report coverage rather than silently
    presenting an incomplete comparison as a full-trajectory score.
    """
    if telemetry is None:
        return []
    curves = [curve for curve in integrated_telemetry_curves(
        telemetry, primary_group, [s for s, _, _ in cases], telemetry_shift) if curve[0] < 2]
    results = []
    for solution, _, label in cases:
        histories = paper_histories(solution)
        losses, matched, observed = {}, 0, 0
        for index, time, values, phase_label, _ in curves:
            phase = 2 if 'Booster return' in phase_label else 1 if 'Stage 2' in phase_label else 0
            time, values = np.asarray(time), np.asarray(values)
            finite = np.isfinite(time) & np.isfinite(values)
            observed += int(finite.sum())
            predicted = np.full(time.size, np.nan)
            for group, sim_time, altitude, speed in histories:
                if group != phase:
                    continue
                unique, positions = np.unique(sim_time, return_index=True)
                sim_values = (altitude, speed)[index][positions]
                inside = finite & (time >= unique[0]) & (time <= unique[-1])
                predicted[inside] = np.interp(time[inside], unique, sim_values)
            valid = finite & np.isfinite(predicted)
            matched += int(valid.sum())
            if not valid.any():
                continue
            # Altitudes in the figure/telemetry are km; scales are expressed in SI.
            residual = (predicted[valid] - values[valid]) * (1000. if index == 0 else 1.)
            residual /= (altitude_scale, speed_scale)[index]
            losses.setdefault(phase, []).append((float(np.mean(2*(np.hypot(1., residual)-1.))),
                                                float(np.mean(residual**2))))
        mean = np.mean([np.mean(channels, axis=0) for channels in losses.values()], axis=0) if losses else (None, None)
        results.append(dict(case=label, robust=mean[0], squared=mean[1], matched=matched, observed=observed))
    return results
