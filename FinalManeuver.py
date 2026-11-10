"""Smooth parking-orbit geometry and an explicit impulsive deployment maneuver."""
import casadi as ca
import numpy as np


def orbit_geometry(model, state):
    position, velocity = state[:3], state[3:6]
    radius = ca.norm_2(position)
    momentum = ca.cross(position, velocity)
    momentum_norm = ca.norm_2(momentum)
    radial = position / radius
    normal = momentum / momentum_norm
    transverse = ca.cross(normal, radial)
    eccentricity = ca.cross(velocity, momentum) / model.mu - radial
    inverse_a = 2. / radius - ca.sumsqr(velocity) / model.mu
    return dict(radial=radial, normal=normal, transverse=transverse,
                eccentricity=eccentricity, inverse_a=inverse_a, momentum=momentum,
                momentum_norm=momentum_norm)


def add_final_maneuver(opti, diagnostics, model, state, target, phase):
    """Keep the parking apogee at target perigee, with free coast angle/RAAN.

    Project the eccentricity vector onto a local orbital basis instead of
    dividing by its norm. At circular parking orbits the coast angle remains a
    free decision variable. All added constraints are smooth at zero burn Δv.
    """
    geometry = orbit_geometry(model, state)
    radius = float(target['perigee'])
    minimum_perigee = model.R0 + model.ParkingOrbit_PerigeeAlt
    coast_angle = opti.variable()
    after_direction = opti.variable(3)
    delta_v_scaled = opti.variable()
    velocity_scale = 1000.  # m/s; keep a small LEO trim burn well scaled.
    diagnostics.add(phase, 'Maneuver coast angle bounds', opti.bounded(-2*np.pi, coast_angle, 2*np.pi), 180. / np.pi, 'deg')
    diagnostics.add(phase, 'Deployment velocity component bounds', opti.bounded(-1., after_direction, 1.))
    burn_radial = (ca.cos(coast_angle) * geometry['radial']
                   + ca.sin(coast_angle) * geometry['transverse'])
    apogee_eccentricity = radius * geometry['inverse_a'] - 1.
    diagnostics.add(phase, 'Parking apogee radial geometry',
                    ca.dot(geometry['eccentricity'], geometry['radial'])
                    + apogee_eccentricity * ca.cos(coast_angle) == 0.)
    diagnostics.add(phase, 'Parking apogee transverse geometry',
                    ca.dot(geometry['eccentricity'], geometry['transverse'])
                    + apogee_eccentricity * ca.sin(coast_angle) == 0.)
    diagnostics.add(phase, 'Parking apogee eccentricity lower bound', apogee_eccentricity >= 0.)
    diagnostics.add(phase, 'Parking perigee lower bound',
                    geometry['inverse_a'] * (.5 * (minimum_perigee + radius)) <= 1.)
    target_a = .5 * (target['apogee'] + target['perigee'])
    target_speed = np.sqrt(model.mu * (2. / radius - 1. / target_a))
    before = geometry['momentum_norm'] / radius * ca.cross(geometry['normal'], burn_radial)
    after = target_speed * after_direction
    diagnostics.add(phase, 'Deployment velocity unit magnitude', ca.sumsqr(after_direction) == 1.)
    diagnostics.add(phase, 'Deployment velocity tangency', ca.dot(burn_radial, after_direction) == 0.)
    diagnostics.add(phase, 'Deployment inclination target',
                    ca.cross(burn_radial, after_direction)[2] == np.cos(target['i']))
    diagnostics.add(phase, 'Maneuver delta-v lower bound', delta_v_scaled >= 0., velocity_scale, 'm/s')
    diagnostics.add(phase, 'Maneuver delta-v upper bound',
                    delta_v_scaled <= (target_speed + geometry['momentum_norm'] / radius + .001) / velocity_scale,
                    velocity_scale, 'm/s')
    # A conservative 1 mm/s norm floor keeps Hessians finite at a zero burn.
    # The equality avoids the poorly conditioned apex of a squared cone bound.
    diagnostics.add(phase, 'Maneuver delta-v magnitude',
                    delta_v_scaled == ca.sqrt(ca.sumsqr((after - before) / velocity_scale) + 1e-12),
                    velocity_scale, 'm/s')
    delta_v = velocity_scale * delta_v_scaled
    retained = ca.exp(-delta_v / (.99 * model.SecondStage_Vac_Isp * model.g0))
    return dict(reserve=state[6] * (1. - retained), delta_v=delta_v,
                position=radius * burn_radial, before=before, after=after,
                coast_angle=coast_angle, after_direction=after_direction,
                delta_v_scaled=delta_v_scaled, velocity_scale=velocity_scale,
                variables=[coast_angle, after_direction, delta_v_scaled])


def seed_final_maneuver(opti, maneuver, model, state, target, saved=None):
    """Numeric initialization only; no piecewise expressions enter the NLP."""
    state = np.asarray(state, dtype=float).reshape(-1)
    geometry = {key: np.asarray(value, dtype=float).reshape(-1)
                for key, value in orbit_geometry(model, ca.DM(state)).items()}
    radial, tangent, normal = (geometry[key] for key in ('radial', 'transverse', 'normal'))
    eccentricity = geometry['eccentricity']
    if np.linalg.norm(eccentricity) > 1e-8:
        angle = np.arctan2(-np.dot(eccentricity, tangent), -np.dot(eccentricity, radial))
    else:
        node = np.cross([0., 0., 1.], normal)
        node = node / np.linalg.norm(node) if np.linalg.norm(node) > 1e-10 else radial
        angle = np.arctan2(np.dot(node, tangent), np.dot(node, radial))
    burn_radial = np.cos(angle) * radial + np.sin(angle) * tangent
    # Project the desired normal into the plane perpendicular to burn radius.
    vertical = np.array([0., 0., 1.]) - burn_radial[2] * burn_radial
    rho2 = np.dot(vertical, vertical)
    horizontal = np.cross(burn_radial, [0., 0., 1.])
    if np.linalg.norm(horizontal) < 1e-10:
        horizontal = normal.copy()
    horizontal /= np.linalg.norm(horizontal)
    if np.dot(horizontal, normal) < 0:
        horizontal *= -1
    coefficient = np.cos(target['i']) / max(rho2, 1e-12)
    target_normal = coefficient * vertical + np.sqrt(max(0., 1. - coefficient**2 * rho2)) * horizontal
    if np.linalg.norm(target_normal) < 1e-10:
        target_normal = normal
    target_normal /= np.linalg.norm(target_normal)
    direction = np.cross(target_normal, burn_radial)
    direction /= np.linalg.norm(direction)
    if saved is not None and 'final_burn_coast_angle' in saved and 'final_burn_velocity_after' in saved:
        previous_angle = float(saved['final_burn_coast_angle'])
        previous_velocity = np.asarray(saved['final_burn_velocity_after'], dtype=float).reshape(3)
        if (np.isfinite(previous_angle) and abs(previous_angle) <= 2*np.pi
                and np.isfinite(previous_velocity).all() and np.linalg.norm(previous_velocity) > 0):
            angle = previous_angle
            burn_radial = np.cos(angle) * radial + np.sin(angle) * tangent
            direction = previous_velocity / np.linalg.norm(previous_velocity)
    radius = target['perigee']
    speed = np.sqrt(model.mu * (2. / radius - 2. / (target['apogee'] + radius)))
    before = float(geometry['momentum_norm'][0]) / radius * np.cross(normal, burn_radial)
    delta_v = np.linalg.norm(speed * direction - before)
    opti.set_initial(maneuver['coast_angle'], angle)
    opti.set_initial(maneuver['after_direction'], direction)
    opti.set_initial(maneuver['delta_v_scaled'], max(delta_v, .01) / maneuver['velocity_scale'])
