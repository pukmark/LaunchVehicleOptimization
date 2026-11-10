"""Combined flight histories from SI optimization results, including coasts."""
import numpy as np

from LV_Type_scaled import DispesrionFactorsType, LaunchVehicle_ECI, VLType
from TelemetryFrames import eci_to_ecef_velocity
from PlotLegends import set_vehicle_legend


def vector_angle(velocity, thrust):
    denominator = np.linalg.norm(velocity, axis=0) * np.linalg.norm(thrust, axis=0)
    cosine = np.divide(np.sum(velocity * thrust, axis=0), denominator,
                       out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))


def second_stage_coast(solution):
    """Reconstruct the same separation-to-ignition propagation as the optimizer."""
    start, end = float(np.ravel(solution['t1_vec'])[-1]), float(np.ravel(solution['t2_vec'])[0])
    if end <= start:
        return None
    model = LaunchVehicle_ECI(DispesrionFactorsType(**solution.get('dispersion', {})),
                              solution.get('configuration', 1), launch_site=solution.get('launch_site'))
    x1 = np.asarray(solution['x1'])[:, -1]
    position, velocity = model.local_to_eci_func([x1[0], 0., x1[1]], [x1[2], 0., x1[3]],
                                                solution['LaunchAz'])
    initial = np.r_[position.full().ravel(), velocity.full().ravel(),
                    model.SecondStage_FullMass + float(solution['payload_mass'])]
    time = np.linspace(start, end, 26)
    state = np.column_stack([
        model.unscale_x(model.dynamics_kp1(model.scale_x(initial), np.full(model.nu, 1e-8),
                                          elapsed / model.scaleT)).full().ravel()
        for elapsed in time - start])
    altitude = np.array([float(model.eci_to_alt(state[:, k])) for k in range(len(time))])
    return time, state, altitude


def return_controls(solution, model):
    """Recover engine histories from burn durations and SI thrust controls.

    Return thrust is expressed relative to the model's three-engine reference;
    the boostback model uses vacuum thrust, reentry/landing use sea-level thrust.
    Angle history is shown only during boostback.
    """
    start = float(np.ravel(solution['t4_vec'])[0])
    times, throttles, angles = [], [], []

    def interval(duration, throttle, angle):
        nonlocal start
        finish = start + float(duration)
        times.extend((start, finish))
        throttles.extend((throttle, throttle))
        angles.extend(np.broadcast_to(angle, (2,)))
        start = finish

    if 'dt4_boostback' in solution:
        coast = float(solution.get('dt4_coast_after_sep', 0.))
        if coast > 0:
            interval(coast, 0., np.nan)
        thrust = np.asarray(solution['u4_boostback']).ravel()
        end = start + float(solution['dt4_boostback'])
        t = np.asarray(solution['t4_vec']).ravel()
        x = np.asarray(solution['x4'])
        # Keep the first occurrence at boundaries where samples are duplicated.
        unique, indices = np.unique(t, return_index=True)
        velocity = np.vstack([np.interp([start, end], unique, x[indices, k]) for k in (2, 3)])
        interval(solution['dt4_boostback'], np.linalg.norm(thrust) / (model.FirstStage_Vac_Thrust / 3.),
                 vector_angle(velocity, np.broadcast_to(thrust[:, None], velocity.shape)))
    interval(solution['dt4_ballistic'], 0., np.nan)
    interval(solution['dt4_reentry'], float(solution['u4_reentry']) / (model.FirstStage_SL_Thrust / 3.), np.nan)
    interval(solution['dt4_before_landing'], 0., np.nan)
    landing = np.asarray(solution['u4_landing']).ravel() / (model.FirstStage_SL_Thrust / 3.)
    landing_time = start + np.arange(len(landing) + 1) * float(solution['dt4_landing'])
    times.extend(landing_time)
    throttles.extend(np.r_[landing, 0.])
    if 'landing_thrust_angle' in solution:
        states = np.asarray(solution['x4_landing'])
        angle = np.asarray(solution['landing_thrust_angle']).ravel()
        radial_angle = np.arctan2(states[0, :-1], model.R0 + states[1, :-1])
        thrust_direction = np.array([np.sin(radial_angle + angle), np.cos(radial_angle + angle)])
        angles.extend(np.r_[vector_angle(states[2:4, :-1], thrust_direction), np.nan])
    else:
        angles.extend(np.full(len(landing_time), np.nan))
    return np.asarray(times), np.asarray(throttles), np.asarray(angles)


def integrated_telemetry_curves(telemetry, primary_group, solutions, shift=0.):
    """Route measured channels without truncating explicit phase telemetry."""
    time = np.asarray(telemetry.time).ravel() + shift
    shared = primary_group == 'Stage 1 / booster' and bool(telemetry.groups['primary'])
    windows = {
        'stage1': [(min(0., float(np.min(s['t1_vec']))), float(np.max(s['t1_vec']))) for s in solutions],
        'booster': [(float(np.min(s['t4_vec'])), float(np.max(s['t4_vec']))) for s in solutions if 't4_vec' in s],
    }
    boostback = []
    for s in solutions:
        if 't4_vec' in s and 'dt4_boostback' in s:
            start = float(np.min(s['t4_vec'])) + float(s.get('dt4_coast_after_sep', 0.))
            boostback.append((start, start + float(s['dt4_boostback'])))

    def within(intervals):
        mask = np.zeros(time.size, dtype=bool)
        for start, end in intervals:
            mask |= (time >= start) & (time <= end)
        return mask

    for group, label in (('stage1', 'Stage 1'), ('stage2', 'Stage 2'), ('booster', 'Booster return')):
        explicit = telemetry.channels(group, 'Phase columns')
        routed = telemetry.channels(group, primary_group)
        if shared and group in windows:
            # A shared broadcast feed needs simulation windows to distinguish
            # ascent from return. Without results, show that feed just once.
            if not solutions:
                if group == 'booster':
                    routed = explicit
                else:
                    label = 'Stage 1 / booster'
            else:
                mask = within(windows[group])
                routed = {key: np.where(mask, values, explicit.get(key, np.full(time.size, np.nan)))
                          for key, values in routed.items()}
        for index, key, scale in ((0, 'altitude', 1.), (1, 'speed', 1.), (2, 'mass', 1.),
                                   (3, 'thrust_factor', 100.), (4, 'alpha', 1.), (4, 'thrust_angle', 1.)):
            if key not in routed:
                continue
            values = np.asarray(routed[key], dtype=float).ravel() * scale
            if group == 'booster' and index == 4:
                values = np.where(within(boostback), values, np.nan)
            if not (np.isfinite(time) & np.isfinite(values)).any():
                continue
            suffix = ' · thrust/velocity angle' if key == 'thrust_angle' else ''
            yield index, time, values, f'Telemetry · {label}{suffix}', '--' if group == 'booster' else ':'


def draw_integrated_results(figure, cases, *, telemetry=None, primary_group='Phase columns',
                            telemetry_color='#CC79A7', telemetry_shift=0.):
    """Draw all visible cases; each tuple is (solution, color, case label)."""
    figure.clear()
    curves = list(integrated_telemetry_curves(telemetry, primary_group, [s for s, _, _ in cases],
                                             telemetry_shift)) if telemetry is not None else []
    if not cases and not curves:
        figure.text(.5, .5, 'No displayed results or matching telemetry.', ha='center', va='center', color='#666666')
        return False
    grid = figure.add_gridspec(2, 6)
    axes = [figure.add_subplot(grid[0, :2])]
    axes += [figure.add_subplot(grid[0, 2:4], sharex=axes[0]),
             figure.add_subplot(grid[0, 4:], sharex=axes[0]),
             figure.add_subplot(grid[1, :3], sharex=axes[0]),
             figure.add_subplot(grid[1, 3:], sharex=axes[0])]

    legend_phases = [set() for _ in axes]
    for s, color, label in cases:
        model = VLType()
        model.ApplyVehicleConfiguration(s.get('configuration', 1))
        model.ApplyScenarioDispersionToLV(DispesrionFactorsType(**s.get('dispersion', {})))

        def line(index, time, values, phase, style='-', control=False):
            if np.isfinite(values).any():
                legend_phases[index].add('Booster' if phase.startswith('Booster') else 'Main vehicle')
            axes[index].plot(np.asarray(time).ravel(), np.asarray(values).ravel(),
                             color=color, linestyle=style, linewidth=1.6,
                             label=f'{label} · {phase}', drawstyle='steps-post' if control else 'default')

        t, x, u = np.ravel(s['t1_vec']), np.asarray(s['x1']), np.asarray(s['u1'])
        for index, values in enumerate((np.ravel(s['alt1_sol']) / 1000.,
                                        np.linalg.norm(x[2:4], axis=0), x[4] / 1000.)):
            line(index, t, values, 'Stage 1')
        line(3, t, np.r_[u[0], u[0, -1]] * 100., 'Stage 1', control=True)
        line(4, t, np.rad2deg(np.r_[u[1], u[1, -1]]), 'Stage 1', control=True)

        coast = second_stage_coast(s)
        if coast is not None:
            tc, xc, altitude = coast
            for index, values in enumerate((altitude / 1000.,
                    np.linalg.norm(eci_to_ecef_velocity(xc[:3], xc[3:6], tc), axis=0), xc[6] / 1000.)):
                line(index, tc, values, 'Stage 2 coast')
            line(3, tc, np.zeros(len(tc)), 'Stage 2 coast', control=True)

        for number in (2, 3):
            t, x, u = np.ravel(s[f't{number}_vec']), np.asarray(s[f'x{number}']), np.asarray(s[f'u{number}'])
            relative = x[3:6] - np.cross([0., 0., model.omega_earth], x[:3].T).T
            for index, values in enumerate((np.ravel(s[f'alt{number}_sol']) / 1000.,
                                            np.linalg.norm(relative, axis=0), x[6] / 1000.)):
                line(index, t, values, 'Stage 2')
            throttle = np.linalg.norm(u, axis=0) / model.SecondStage_Thrust * 100.
            angle = vector_angle(relative[:, :-1], u)
            line(3, t, np.r_[throttle, throttle[-1]], 'Stage 2', control=True)
            line(4, t, np.r_[angle, angle[-1]], 'Stage 2', control=True)

        if 'x4' in s:
            t, x = np.ravel(s['t4_vec']), np.asarray(s['x4'])
            for index, values in enumerate((np.ravel(s['alt4_sol']) / 1000.,
                                            np.linalg.norm(x[:, 2:4], axis=1), x[:, 4] / 1000.)):
                line(index, t, values, 'Booster return', '--')
            time, throttle, angle = return_controls(s, model)
            line(3, time, throttle * 100., 'Booster return', '--', control=True)
            if np.isfinite(angle).any():
                line(4, time, angle, 'Booster return · thrust direction', '--', control=True)

    for index, time, values, label, style in curves:
        legend_phases[index].add('Booster' if 'Booster return' in label else 'Main vehicle')
        axes[index].plot(time, values, color=telemetry_color, linestyle=style,
                         marker='.', markersize=3, linewidth=1.2, label=label)

    for axis, title, unit in zip(axes, ('Altitude', 'Velocity (Earth-fixed)', 'Mass', 'Throttle',
                                       'AoA / thrust–velocity angle'),
                                 ('Altitude [km]', 'Speed [m/s]', 'Mass [t]', 'Throttle [%]', 'Angle [deg]')):
        axis.set(title=title, xlabel='Time since launch [s]', ylabel=unit)
        axis.grid(True, alpha=.25)
        axis.tick_params(labelsize=8)
        set_vehicle_legend(axis, legend_phases[axes.index(axis)], fontsize=7, loc='best')
    figure.suptitle('Integrated flight results', fontsize=12)
    figure.supxlabel('S2/boostback angle: thrust vs. air-relative velocity; booster angle shown only during boostback. '
                     'Throttle uses each phase’s reference thrust.', fontsize=8)
    return bool(curves)
