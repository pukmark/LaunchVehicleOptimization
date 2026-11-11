"""Check illustration timing and frames without solving a trajectory."""
import unittest
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from LV_Type_scaled import VLType
from Trajectory3D import second_burn_history, maneuver_summary_text
from OptimizationGUI import draw_phase
from IntegratedResults import draw_integrated_results
from TelemetryFrames import eci_to_ecef_velocity


def burn_case():
    model = VLType()
    radius, apogee = model.R0 + 150000, model.R0 + 500000
    a = (radius + apogee) / 2
    initial_speed = np.sqrt(model.mu * (2 / radius - 1 / a))
    before_speed = np.sqrt(model.mu * (2 / apogee - 1 / a))
    mass = 100000.
    fuel = 50 * model.SecondStage_Thrust / (.99 * model.SecondStage_Vac_Isp * model.g0)
    dv = .99 * model.SecondStage_Vac_Isp * model.g0 * np.log(mass / (mass - fuel))
    state = np.repeat(np.array([radius, 0, 0, 0, initial_speed, 0, mass])[:, None], 3, axis=1)
    solution = dict(x3=state, final_burn_position=np.array([-apogee, 0., 0.]),
                    final_burn_velocity_before=np.array([0., -before_speed, 0.]),
                    final_burn_velocity_after=np.array([0., -before_speed - dv, 0.]),
                    propellant_mass_for_final_dv=fuel, payload_mass=10000., LaunchAz=0.)
    for number, times in ((2, [200., 300., 400.]), (3, [400., 500., 600.])):
        solution[f'x{number}'] = state.copy()
        solution[f't{number}_vec'] = np.array(times)
        solution[f'u{number}'] = np.tile([0., model.SecondStage_Thrust, 0.], (2, 1)).T
        solution[f'alt{number}_sol'] = np.full(3, radius - model.R0)
        solution[f'i{number}_sol'] = np.zeros(3)
        solution[f'apogee{number}_sol'] = np.full(3, apogee)
        solution[f'perigee{number}_sol'] = np.full(3, radius)
        solution[f'Isp{number}_sol'] = np.full(2, model.SecondStage_Vac_Isp)
    solution.update(t1_vec=np.array([0., 100., 200.]), x1=np.ones((5, 3)),
                    u1=np.ones((2, 2)), alt1_sol=np.zeros(3))
    return solution, dict(apogee=apogee, perigee=radius, i=0.)


class SecondBurnTests(unittest.TestCase):
    def test_duration_and_physical_time_are_distinct(self):
        solution, target = burn_case()
        original = solution['x3'].copy()
        burn = second_burn_history(solution, target)
        dv = np.linalg.norm(solution['final_burn_velocity_after'] - solution['final_burn_velocity_before'])
        self.assertAlmostEqual(burn['duration'], dv / VLType().Payload_Max_acc)
        np.testing.assert_allclose(burn['time'][[0, -1]], [700., 700. + burn['duration']])
        acceleration = np.diff(burn['state'][3:6], axis=1) / np.diff(burn['time'])
        np.testing.assert_allclose(np.linalg.norm(acceleration, axis=0), VLType().Payload_Max_acc)
        np.testing.assert_allclose(np.linalg.norm(burn['thrust'], axis=0) / burn['state'][6],
                                   VLType().Payload_Max_acc)
        self.assertAlmostEqual(burn['state'][6, -1], original[6, -1] -
                               solution['propellant_mass_for_final_dv'])
        a = (VLType.R0 + 150000 + VLType.R0 + 500000) / 2
        self.assertAlmostEqual(burn['physical_start'], 600 + np.pi * np.sqrt(a**3 / VLType.mu))
        np.testing.assert_allclose(burn['state'][3:6, 0], solution['final_burn_velocity_before'])
        np.testing.assert_allclose(burn['state'][3:6, -1], solution['final_burn_velocity_after'])
        np.testing.assert_array_equal(solution['x3'], original)
        text = maneuver_summary_text(solution, target)
        self.assertIn(f"length {burn['duration']:.1f} s", text)
        self.assertIn(f"start T+{burn['physical_start'] / 60:.2f} min", text)
        self.assertNotIn('velocity (ECI)', text)

    def test_plots_include_separate_burn_with_correct_frame(self):
        solution, target = burn_case()
        burn = second_burn_history(solution, target)
        for frame in ('Inertial', 'ECEF'):
            figure = Figure(figsize=(14, 8), constrained_layout=True)
            draw_phase(figure, 'Second stage', solution, target, speed_frame=frame)
            curve = next(line for line in figure.axes[1].lines if 'S2 burn 2 (illustration)' in line.get_label())
            np.testing.assert_allclose(curve.get_xdata()[[0, -1]], [700, 700 + burn['duration']])
            velocity = burn['state'][3:6]
            if frame == 'ECEF':
                velocity = eci_to_ecef_velocity(burn['state'][:3], velocity,
                                               burn['physical_start'] + burn['time'] - 700)
            np.testing.assert_allclose(curve.get_ydata(), np.linalg.norm(velocity, axis=0))
            note = figure.axes[1].texts[0].get_text()
            self.assertEqual(note, f"Case · Start T+{burn['physical_start'] / 60:.2f} min · Length {burn['duration']:.1f} s")
            FigureCanvasAgg(figure).draw()
        figure = Figure(figsize=(14, 8), constrained_layout=True)
        draw_integrated_results(figure, [(solution, 'blue', 'Case 1')])
        curve = next(line for line in figure.axes[1].lines if 'burn 2 (illustration)' in line.get_label())
        np.testing.assert_allclose(curve.get_xdata()[[0, -1]], [700, 700 + burn['duration']])
        FigureCanvasAgg(figure).draw()

    def test_direction_only_pulse_has_duration_and_preserves_true_speeds(self):
        solution, target = burn_case()
        before = solution['final_burn_velocity_before']
        angle = np.deg2rad(20.)
        # Rotate about the burn radius: identical speeds, different orbit planes.
        after = np.array([before[0], before[1] * np.cos(angle), before[1] * np.sin(angle)])
        solution['final_burn_velocity_after'] = after
        delta_v = np.linalg.norm(after - before)
        model = VLType()
        solution['propellant_mass_for_final_dv'] = solution['x3'][6, -1] * (
            -np.expm1(-delta_v / (.99 * model.SecondStage_Vac_Isp * model.g0)))
        burn = second_burn_history(solution, target)
        self.assertGreater(burn['duration'], 0.)
        self.assertAlmostEqual(burn['duration'], delta_v / model.Payload_Max_acc)
        np.testing.assert_allclose(burn['state'][3:6, 0], before)
        np.testing.assert_allclose(burn['state'][3:6, -1], after)
        np.testing.assert_allclose(np.linalg.norm(burn['state'][3:6, [0, -1]], axis=0),
                                   np.linalg.norm(before))

    def test_uses_saved_custom_payload_limit(self):
        from VehicleDefinitions import vehicle_editor_values, vehicle_from_editor
        solution, target = burn_case()
        values = vehicle_editor_values(1)
        durations = []
        for limit in (20., 40.):
            values['Payload_Max_acc'] = str(limit)
            solution['configuration'] = vehicle_from_editor('Payload limit case', values)
            burn = second_burn_history(solution, target)
            durations.append(burn['duration'])
            acceleration = np.diff(burn['state'][3:6], axis=1) / np.diff(burn['time'])
            np.testing.assert_allclose(np.linalg.norm(acceleration, axis=0), limit)
        self.assertAlmostEqual(durations[0], 2 * durations[1])

    def test_missing_and_direct_injection_have_no_burn(self):
        self.assertIsNone(second_burn_history({}))
        solution, target = burn_case()
        solution['direct_injection'] = True
        self.assertIsNone(second_burn_history(solution, target))
        del solution['direct_injection']
        del solution['propellant_mass_for_final_dv']
        self.assertIsNone(second_burn_history(solution, target))
