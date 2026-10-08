"""ECI trajectory and bounded two-body orbit rendering for the desktop GUI."""
import numpy as np
from textwrap import shorten

from LV_Type_scaled import VLType
from LaunchSites import parse_launch_site
from TelemetryFrames import local_to_ecef_velocity
from VehicleDefinitions import vehicle_name


def orbit_curve(position, velocity, mu=VLType.mu, samples=361):
    """Sample a bound osculating Kepler ellipse, including the closing point."""
    position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,) or not np.isfinite([position, velocity]).all():
        raise ValueError('Orbit position and velocity must be finite three-component vectors.')
    radius = np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    momentum = np.linalg.norm(angular_momentum)
    if radius <= 0 or momentum <= 0 or not np.isfinite(mu) or mu <= 0:
        raise ValueError('Orbit requires a positive radius, gravity, and angular momentum.')
    eccentricity_vector = np.cross(velocity, angular_momentum) / mu - position / radius
    eccentricity = np.linalg.norm(eccentricity_vector)
    energy = np.dot(velocity, velocity) / 2 - mu / radius
    if not np.isfinite(energy) or energy >= 0 or eccentricity >= 1:
        raise ValueError('The final state does not define a bound orbit.')
    semimajor = -mu / (2 * energy)
    normal = angular_momentum / momentum
    periapsis_direction = eccentricity_vector / eccentricity if eccentricity > 1e-10 else position / radius
    transverse = np.cross(normal, periapsis_direction)
    anomaly = np.linspace(0, 2*np.pi, samples)
    radii = momentum**2 / mu / (1 + eccentricity * np.cos(anomaly))
    points = radii[:, None] * (np.cos(anomaly)[:, None] * periapsis_direction
                               + np.sin(anomaly)[:, None] * transverse)
    apogee = semimajor * (1 + eccentricity)
    perigee = semimajor * (1 - eccentricity)
    return {'points': points, 'apogee': float(apogee), 'perigee': float(perigee),
            'apogee_position': -apogee * periapsis_direction, 'normal': normal}


def solution_orbits(solution, target, mu=VLType.mu):
    """Match Main.py: parking orbit, then the saved speed change at apogee."""
    state = np.asarray(solution['x3'], dtype=float)[:, -1]
    parking = orbit_curve(state[:3], state[3:6], mu)
    position = parking['apogee_position']
    if 'v3_desired' in solution:
        speed = float(np.asarray(solution['v3_desired']).item())
    else:
        semimajor = (target['apogee'] + target['perigee']) / 2
        speed_squared = mu * (2 / parking['apogee'] - 1 / semimajor)
        if speed_squared <= 0:
            raise ValueError('The target orbit is incompatible with the parking apogee.')
        speed = np.sqrt(speed_squared)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('The final-orbit speed must be finite and positive.')
    direction = np.cross(parking['normal'], position / np.linalg.norm(position))
    final = orbit_curve(position, speed * direction, mu)
    return parking, final


def first_stage_eci_positions(solution):
    """Use the solver's launch-local position transform, including its surface origin.

    Local Z already contains launch altitude and its dispersion. Like the model's
    stage-separation transform, the launch basis is fixed at the initial ECI epoch.
    """
    model = VLType()
    model.ApplyVehicleConfiguration(solution.get('configuration', 1))
    site = parse_launch_site(solution.get('launch_site'))
    latitude, longitude = np.deg2rad([site['latitude'], site['longitude']])
    radius = model.R0 / np.sqrt(1 - model.e**2 * np.sin(latitude)**2)
    origin = radius * np.array([np.cos(latitude)*np.cos(longitude),
                                np.cos(latitude)*np.sin(longitude), (1-model.e**2)*np.sin(latitude)])
    state = np.asarray(solution['x1'])
    local = np.vstack((state[0], np.zeros(state.shape[1]), state[1]))
    positions = local_to_ecef_velocity(local, float(np.asarray(solution['LaunchAz']).item()),
                                       site['latitude'], site['longitude'])
    return (positions + origin[:, None]).T


def case_results_text(solution, label, parking=None, final=None):
    """Summarize computed case metadata, leaving unavailable results explicit."""
    def formatted(value, unit, scale=1., digits=1):
        if value is None:
            return '—'
        value = float(np.asarray(value).item()) * scale
        return f'{value:,.{digits}f} {unit}'.strip() if np.isfinite(value) else '—'

    recovery = 'RTLS' if 'x4_boostback' in solution else 'ASDS' if 'x4' in solution else 'EXP'
    lines = [f'{label} · {recovery}', shorten(vehicle_name(solution.get('configuration', 1)), width=35, placeholder='…'),
             f"Payload: {formatted(solution.get('payload_mass'), 't', .001, 3)}",
             f"Azimuth: {formatted(solution.get('LaunchAz'), '° E of N', 180/np.pi, 2)}"]
    if parking is not None and final is not None:
        model = VLType()
        model.ApplyVehicleConfiguration(solution.get('configuration', 1))
        radius = parking['apogee']
        a_parking = (parking['apogee'] + parking['perigee']) / 2
        a_final = (final['apogee'] + final['perigee']) / 2
        delta_v = abs(np.sqrt(model.mu*(2/radius - 1/a_final))
                      - np.sqrt(model.mu*(2/radius - 1/a_parking)))
        lines.append(f'Δv at parking apogee: {delta_v:,.2f} m/s')
        for orbit, name in ((parking, 'Parking'), (final, 'Final')):
            apogee, perigee = ((orbit[key] - model.R0) / 1000 for key in ('apogee', 'perigee'))
            lines.append(f'{name}: {apogee:,.1f} × {perigee:,.1f} km')
        inclination = np.rad2deg(np.arccos(np.clip(parking['normal'][2], -1, 1)))
        lines.append(f'Inclination: {inclination:.2f}°')
    else:
        lines += ['Δv / parking / final orbit: —']
    for number, name in ((1, 'MECO'), (3, 'SECO')):
        if f't{number}_vec' in solution:
            lines.append(f"{name}: {formatted(np.asarray(solution[f't{number}_vec']).ravel()[-1], 's')}")
    if 'Qdyn1_sol' in solution:
        pressure = np.asarray(solution['Qdyn1_sol'])
        pressure = pressure[np.isfinite(pressure)]
        lines.append(f"Max-Q (S1): {formatted(pressure.max() if pressure.size else None, 'kPa', .001)}")
    if 'propellant_mass_for_final_dv' in solution:
        lines.append(f"Orbit-burn propellant: {formatted(solution['propellant_mass_for_final_dv'], 't', .001, 3)}")
    return '\n'.join(lines)


def draw_trajectory_3d(figure, cases):
    """Draw visible (solution, target, color, label) cases like Main.py's ax_3d."""
    view = None
    if figure.axes and figure.axes[0].name == '3d':
        view = (figure.axes[0].elev, figure.axes[0].azim)
    figure.clear()
    grid = figure.add_gridspec(3, 4, width_ratios=(1, 1, .95, 1.25))
    axis = figure.add_subplot(grid[:, :2], projection='3d')
    projections = [figure.add_subplot(grid[row, 2]) for row in range(3)]
    if view:
        axis.view_init(*view)
    model = VLType()
    if cases:
        model.ApplyVehicleConfiguration(cases[0][0].get('configuration', 1))
    radius = model.R0 / 1000
    phi, theta = np.mgrid[0:np.pi:25j, 0:2*np.pi:49j]
    axis.plot_surface(radius*np.sin(phi)*np.cos(theta), radius*np.sin(phi)*np.sin(theta),
                      radius*np.cos(phi), color='tab:blue', alpha=.12, linewidth=0, antialiased=False)
    plane_angle, plane_radius = np.meshgrid(np.linspace(0, 2*np.pi, 49), np.linspace(0, radius*1.1, 3))
    axis.plot_surface(plane_radius*np.cos(plane_angle), plane_radius*np.sin(plane_angle),
                      np.zeros_like(plane_radius), color='gray', alpha=.15, linewidth=0)
    for vector, color, label in (([1.3*radius, 0, 0], 'r', 'X'),
                                 ([0, 1.3*radius, 0], 'g', 'Y'), ([0, 0, 1.3*radius], 'b', 'Z')):
        axis.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=.07)
        axis.text(*vector, label, color=color, fontsize=8)
    angles = np.linspace(0, 2*np.pi, 121)
    for projection, (horizontal, vertical) in zip(projections, (('X', 'Y'), ('Y', 'Z'), ('X', 'Z'))):
        projection.plot(radius*np.cos(angles), radius*np.sin(angles), color='gray', alpha=.5, linewidth=1)
        projection.set(xlabel=f'{horizontal} [km]', ylabel=f'{vertical} [km]')
        projection.set_aspect('equal', adjustable='box')
        projection.tick_params(labelsize=7)
        projection.grid(True, alpha=.25)

    limit = 1.35 * radius
    notes = []
    def path(points, color, label, style='-'):
        nonlocal limit
        points = np.asarray(points) / 1000
        limit = max(limit, float(np.max(np.abs(points))))
        axis.plot(*points.T, color=color, linestyle=style, linewidth=1.5, label=label)
        for projection, (horizontal, vertical) in zip(projections, ((0, 1), (1, 2), (0, 2))):
            projection.plot(points[:, horizontal], points[:, vertical], color=color,
                            linestyle=style, linewidth=1.2)

    for index, (solution, target, color, label) in enumerate(cases):
        if 'x1' in solution:
            trajectory = first_stage_eci_positions(solution)
            path(trajectory, color, f'{label} · Stage 1')
            axis.scatter(*(trajectory[0] / 1000), color=color, marker='^', s=18, depthshade=False)
        trajectory = np.concatenate((np.asarray(solution['x2'])[:3].T,
                                     np.asarray(solution['x3'])[:3].T))
        path(trajectory, color, f'{label} · Stage 2', ':')
        parking = final = None
        try:
            parking, final = solution_orbits(solution, target, model.mu)
        except ValueError as exc:
            notes.append(f'{label}: {exc}')
        else:
            for orbit, name, style in ((parking, 'Parking orbit', '--'), (final, 'Final orbit', '-.')):
                path(orbit['points'], color, f'{label} · {name}', style)
        results = figure.add_subplot(grid[index, 3])
        results.set_axis_off()
        text = case_results_text(solution, label, parking, final)
        height_points = results.get_position().height * figure.get_figheight() * 72
        font_size = np.clip(height_points / (1.45 * len(text.splitlines())), 6, 8)
        results.text(.02, .96, text, transform=results.transAxes, va='top', ha='left',
                     fontsize=font_size, color=color, linespacing=1.25,
                     bbox=dict(boxstyle='round,pad=.6', facecolor='white', edgecolor=color, linewidth=1.4))
    limit *= 1.03
    axis.set(xlabel='ECI X [km]', ylabel='ECI Y [km]', zlabel='ECI Z [km]',
             xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
             title='3D launch trajectory (ECI)')
    axis.set_box_aspect((1, 1, 1))
    axis.tick_params(labelsize=8)
    for projection in projections:
        projection.set(xlim=(-limit, limit), ylim=(-limit, limit))
    if cases:
        axis.legend(fontsize=6.5, loc='upper left', ncol=2)
    else:
        figure.text(.35, .5, 'Run a case and enable Display to see its 3D trajectory.',
                    ha='center', va='center', color='#666666')
    if notes:
        figure.text(.01, .01, '\n'.join(notes), fontsize=7, color='#666666')
