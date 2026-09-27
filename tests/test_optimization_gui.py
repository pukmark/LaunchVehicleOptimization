"""GUI integration checks; display checks skip on headless machines."""
from dataclasses import asdict
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")

import casadi as ca
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from OptimizationGUI import (
    DispesrionFactorsType, OptimizationApp, draw_phase, nominal_request,
    parse_dispersion, parse_target_orbit, run_optimization, worker,
    orbital_diagnostics, thrust_velocity_angle, second_stage_propellant_remaining,
    default_case_inputs, RECOVERY, stage_two_speed_acceleration,
)


class DispersionFormTests(unittest.TestCase):
    def test_defaults_and_fractional_edits(self):
        defaults = asdict(DispesrionFactorsType())
        self.assertEqual(asdict(parse_dispersion(defaults)), defaults)
        edits = {name: str(value) for name, value in defaults.items()}
        edits.update(FirstStageThrust="1.025", FirstStage_EmptyMass="-150.5")
        parsed = parse_dispersion(edits)
        self.assertEqual(parsed.FirstStageThrust, 1.025)
        self.assertEqual(parsed.FirstStage_EmptyMass, -150.5)

    def test_rejects_invalid_values_before_solving(self):
        for name, value in (("FirstStageIsp", ""), ("SecondStageIsp", "NaN"),
                            ("LaunchAltDelta", "inf"), ("FirstStageThrust", "0"),
                            ("AtmosphereDensity", "-1"), ("StagePartitionDelta", "0.99"),
                            ("FirstStage_EmptyMass", "-10000000"),
                            ("SecondStage_PropMass", "-10000000")):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                parse_dispersion(dict(asdict(DispesrionFactorsType()), **{name: value}))

    def test_solver_receives_snapshot_and_nominal_target(self):
        request = nominal_request(asdict(DispesrionFactorsType(FirstStageThrust=1.02)), 2, "RTLS")
        with patch("LV_Optimization_Type.LV_Optimization") as constructor:
            run_optimization(request, None)
        self.assertEqual(constructor.call_args.args[0].FirstStageThrust, 1.02)
        self.assertEqual(constructor.call_args.args[1], 2)
        arguments = constructor.return_value.SolveOptimiztion.call_args.kwargs
        self.assertEqual(arguments["RecoveryStrategy"], "RTLS")
        self.assertEqual(arguments["Target_Orbit"]["apogee"], 6578137.0)
        self.assertEqual(arguments["payload_mass_predefined"], -1)
        self.assertFalse(arguments["Plot_interm"])

    def test_custom_orbit_units_and_solver_handoff(self):
        request = nominal_request(asdict(DispesrionFactorsType()), orbit_values={
            "apogee": "35786", "perigee": "200", "inclination": "51.6"})
        target = request["target_orbit"]
        self.assertEqual(target["apogee"], 42164137.0)
        self.assertEqual(target["perigee"], 6578137.0)
        self.assertAlmostEqual(target["i"], np.deg2rad(51.6))
        with patch("LV_Optimization_Type.LV_Optimization") as constructor:
            run_optimization(request, None)
        self.assertEqual(constructor.return_value.SolveOptimiztion.call_args.kwargs["Target_Orbit"], target)

    def test_invalid_orbits_are_rejected(self):
        defaults = {"apogee": "200", "perigee": "200", "inclination": "28.6"}
        for name, value in (("apogee", ""), ("perigee", "NaN"),
                            ("inclination", "inf"), ("perigee", "0"),
                            ("apogee", "-1"), ("apogee", "199"),
                            ("inclination", "-0.1"), ("inclination", "180.1"),
                            ("apogee", "1e308")):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                parse_target_orbit(dict(defaults, **{name: value}))

    def test_payload_mass_forwarding_and_maximize_mode(self):
        for value in ("12500.5", "0", "-1"):
            with self.subTest(value=value):
                request = nominal_request(asdict(DispesrionFactorsType()), payload_mass=value)
                self.assertEqual(request["payload_mass_predefined"], float(value))
                with patch("LV_Optimization_Type.LV_Optimization") as constructor:
                    run_optimization(request, None)
                self.assertEqual(constructor.return_value.SolveOptimiztion.call_args.kwargs[
                    "payload_mass_predefined"], float(value))

    def test_invalid_payload_mass_is_rejected(self):
        for value in ("", "abc", "NaN", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Payload mass"):
                nominal_request(asdict(DispesrionFactorsType()), payload_mass=value)

    def test_worker_persists_failure_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "inputs.json").write_text(json.dumps(nominal_request(asdict(DispesrionFactorsType()))))
            with patch("OptimizationGUI.run_optimization", return_value={"success": False}):
                worker(directory)
            self.assertTrue((directory / "solution_Failed.pickle").exists())
            self.assertFalse((directory / "solution.pickle").exists())


class RemainingPropellantTests(unittest.TestCase):
    def test_payload_separation_subtracts_dispersed_dry_mass_and_final_burn(self):
        for configuration, dry, first_prop, second_prop in ((1, 4000, 395700, 92670),
                                                            (2, 85000, 3250000, 1500000)):
            with self.subTest(configuration=configuration):
                partition_delta, dry_offset = 0.02, 500
                expected_dry = dry * (1 - partition_delta * (first_prop + second_prop) / second_prop) + dry_offset
                state = np.zeros((7, 2))
                state[6, -1] = expected_dry + 12500 + 800 + 1234.5
                result = {'configuration': configuration,
                          'dispersion': asdict(DispesrionFactorsType(StagePartitionDelta=partition_delta,
                                                                    SecondStage_EmptyMass=dry_offset)),
                          'x3': state, 'payload_mass': 12500, 'propellant_mass_for_final_dv': 800}
                self.assertAlmostEqual(second_stage_propellant_remaining(result), 1234.5)
                # Draft or unrelated request settings cannot override saved metadata.
                self.assertAlmostEqual(second_stage_propellant_remaining(result, {'configuration': 1, 'dispersion': {}}), 1234.5)

    def test_missing_final_burn_is_not_reported_as_zero(self):
        self.assertIsNone(second_stage_propellant_remaining({}))


class DiagnosticTests(unittest.TestCase):
    def test_circular_orbit_and_outbound_radial_motion(self):
        mu, radius = 3.986004418e14, 7000000.0
        speed = np.sqrt(mu / radius)
        state = np.array([[radius, radius], [0, 0], [0, 0],
                          [0, 100], [speed, speed], [0, 0], [10000, 10000]])
        result = orbital_diagnostics(state)
        np.testing.assert_allclose(result["radius"], radius)
        np.testing.assert_allclose(result["radial"], [0, 100])
        np.testing.assert_allclose(result["transverse"], speed)
        self.assertAlmostEqual(result["eccentricity"][0], 0)
        self.assertAlmostEqual(result["energy"][0], -mu / (2 * radius))
        self.assertAlmostEqual(result["energy"][1] - result["energy"][0], 5000)
        np.testing.assert_allclose(result["flight_path"], [0, np.rad2deg(np.arctan2(100, speed))])

    def test_thrust_angle_is_independent_of_throttle_and_handles_zero_thrust(self):
        state = np.zeros((7, 5))
        state[3] = 100
        control = np.array([[100, 50, 0, 0], [0, 0, 50, 0], [0, 0, 0, 0]])
        angles = thrust_velocity_angle(state, control)
        np.testing.assert_allclose(angles[:3], [0, 0, 90])
        self.assertTrue(np.isnan(angles[3]))
        state[3:6] = 0
        state[0] = 7000000
        self.assertTrue(np.all(np.isnan(orbital_diagnostics(state)["flight_path"])))


class SimulationAccelerationTests(unittest.TestCase):
    def test_gravity_only_speed_change_is_signed_not_specific_acceleration(self):
        # Radial ballistic ascent in a constant-g test trajectory, with no thrust.
        time = np.array([0., 1., 3.])
        state = np.zeros((7, 3))
        state[0] = 6378137 + 100*time - 0.5*9.8*time**2
        state[3] = 100 - 9.8*time
        state[6] = 1000
        solution = {"t2_vec": time, "x2": state, "acc2_sol": np.zeros(2)}
        np.testing.assert_allclose(stage_two_speed_acceleration(solution, 2),
                                   [np.nan, -9.8, -9.8], equal_nan=True)

    def test_ecef_stationary_trajectory_and_separation_boundary(self):
        from LV_Type_scaled import VLType
        for number, times in ((2, [0., 1., 2.]), (3, [2., 4., 7.])):
            t = np.array(times)
            angle = VLType.omega_earth*t
            radius = VLType.R0 + 200000
            state = np.zeros((7, 3))
            state[0], state[1] = radius*np.cos(angle), radius*np.sin(angle)
            state[3], state[4] = -VLType.omega_earth*state[1], VLType.omega_earth*state[0]
            solution = {f"t{number}_vec": t, f"x{number}": state}
            np.testing.assert_allclose(stage_two_speed_acceleration(solution, number, "ECEF"),
                                       [np.nan, 0., 0.], atol=1e-10, equal_nan=True)


class ResultRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate the real solver output schema using initial iterates so these
        # regression checks do not depend on IPOPT convergence.
        class InitialValues:
            def __init__(self, opti):
                self.opti = opti

            def value(self, expression):
                return self.opti.debug.value(expression, self.opti.initial())

        cls.results, cls.requests = {}, {}
        for recovery in ("EXP", "ASDS", "RTLS"):
            request = nominal_request(asdict(DispesrionFactorsType()), recovery=recovery)
            with patch.object(ca.Opti, "solve", lambda opti: InitialValues(opti)), \
                 patch.object(ca.Opti, "return_status", return_value="test"), \
                 patch.object(ca.Opti, "stats", return_value={"success": True, "iter_count": 0}):
                cls.results[recovery] = run_optimization(request, None)
            cls.requests[recovery] = request

    def test_all_phases_accept_actual_result_shapes_and_units(self):
        for recovery, solution in self.results.items():
            for phase in ("First stage", "S1 details", "Second stage", "S2 propulsion", "S2 orbit", "Booster"):
                if phase == "Booster" and recovery == "EXP":
                    continue
                with self.subTest(recovery=recovery, phase=phase):
                    figure = Figure(figsize=(12, 7), constrained_layout=True)
                    draw_phase(figure, phase, solution, self.requests[recovery]["target_orbit"])
                    FigureCanvasAgg(figure).draw()
                    self.assertGreaterEqual(len(figure.axes), 6)
                    if phase == "Booster":
                        np.testing.assert_allclose(figure.axes[3].lines[0].get_ydata(), solution["x4"][:, 4] / 1000)
                        self.assertEqual(len(figure.axes), 7)
                        self.assertEqual(figure.axes[6].get_ylabel(), "Heat flux [kW/m²]")
                        from Atmosphere_Type import AtmosphereType
                        rho = np.asarray(AtmosphereType.rho_fun(np.asarray(solution["alt4_sol"]).ravel())).ravel()
                        expected = 2e-4 * np.sqrt(rho) * np.linalg.norm(solution["x4"][:, 2:4], axis=1)**3 / 1000
                        np.testing.assert_allclose(figure.axes[6].lines[0].get_ydata(), expected)
                    if phase == "Second stage":
                        np.testing.assert_allclose(figure.axes[3].lines[0].get_ydata(), np.linalg.norm(solution["u2"], axis=0) / 1000)
                    if phase == "S1 details":
                        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), solution["t1_vec"][:-1])
                        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), solution["Isp1_sol"])
                        np.testing.assert_allclose(figure.axes[2].lines[0].get_ydata(), np.rad2deg(solution["u1"][1]))
                    if phase == "S2 propulsion":
                        self.assertEqual(len(figure.axes[0].lines), 2)
                        np.testing.assert_allclose(figure.axes[0].lines[1].get_xdata(), solution["t3_vec"][:-1])
                        np.testing.assert_allclose(figure.axes[3].lines[0].get_ydata(), solution["u2"][0] / 1000)
                    if phase == "S2 orbit":
                        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), np.linalg.norm(solution["x2"][:3], axis=0) / 1000)

    def make_app(self, directory):
        try:
            import tkinter as tk
        except ImportError as exc:
            self.skipTest(str(exc))
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        root.withdraw()
        def cleanup():
            try:
                if not root.winfo_exists():
                    return
            except tk.TclError:
                return
            # Flush queued Matplotlib draws before deleting their Tk commands.
            root.update_idletasks()
            for callback in root.tk.splitlist(root.tk.call('after', 'info')):
                root.after_cancel(callback)
            root.destroy()
        self.addCleanup(cleanup)
        return OptimizationApp(root, output_root=directory)

    def test_close_reopen_restores_cases_telemetry_and_display(self):
        from TelemetryCSV import load_telemetry_csv
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.csv"
            telemetry_path.write_text("time_s,altitude_m,speed_mps\n0,10,10\n10,100,100\n")
            app = self.make_app(directory)
            for index in range(3):
                app.select_case(index)
                app.payload_mass.set(str(12000 + index * 100))
                app.orbit_variables["apogee"].set(str(300 + index * 100))
                app.variables["FirstStageThrust"].set(str(1 + index / 10))
            app.vehicle.set("Starship")
            app.recovery.set(next(key for key, value in RECOVERY.items() if value == "RTLS"))
            app.orbit_variables["inclination"].set("")  # Preserve incomplete edits too.
            case = app.cases[0]
            case.result = self.results["ASDS"]
            case.request = self.requests["ASDS"]
            case.computed_inputs = default_case_inputs()
            case.state = "Complete"
            case.color = "#123456"
            app.case_visible[1].set(False)
            app.telemetry = load_telemetry_csv(telemetry_path)
            app.telemetry_color = "#654321"
            app.telemetry_primary.set("Stage 2")
            app.telemetry_shift = -3.25
            app.telemetry_shift_input.set("-4.")  # Pending entry is distinct from applied shift.
            app.refresh_comparison()
            app.notebook.select(app.tabs["S2 orbit"])
            expected = [deepcopy(item.draft) for item in app.cases]
            app.close()  # Closing must flush without waiting for the debounce timer.
            restored = self.make_app(directory)
            self.assertEqual(restored.selected_case, 2)
            self.assertEqual(restored.capture_inputs(), expected[2])
            self.assertEqual([item.draft for item in restored.cases], expected)
            self.assertEqual(restored.cases[0].color, "#123456")
            self.assertEqual([item.get() for item in restored.case_visible], [True, False, True])
            np.testing.assert_array_equal(restored.cases[0].result["x2"], case.result["x2"])
            self.assertEqual(restored.cases[0].request, case.request)
            self.assertTrue(restored.cases[0].edited)
            self.assertEqual(restored.telemetry.path, telemetry_path)
            self.assertEqual(restored.telemetry_color, "#654321")
            self.assertEqual(restored.telemetry_primary.get(), "Stage 2")
            self.assertEqual(restored.telemetry_shift, -3.25)
            self.assertEqual(restored.telemetry_shift_input.get(), "-4.")
            self.assertEqual(restored.notebook.tab(restored.notebook.select(), "text"), "S2 orbit")
            self.assertTrue(restored.figures["Second stage"].axes[0].lines)
            self.assertEqual(restored.notebook.tab(restored.tabs["Booster"], "state"), "normal")

    def test_reset_gui_persists_defaults_and_keeps_run_files(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            history = Path(directory) / "previous_run"
            history.mkdir()
            artifact = history / "inputs.json"
            artifact.write_text('{"payload_mass_predefined": 12500}')
            app.select_case(1)
            app.payload_mass.set("12500")
            app.cases[0].result = self.results["ASDS"]
            app.cases[0].request = self.requests["ASDS"]
            app.cases[0].computed_inputs = default_case_inputs()
            app.case_visible[0].set(False)
            app.telemetry_shift = 42
            app.telemetry_visible.set(False)
            app.reset_gui_button.invoke()
            self.assertTrue(artifact.exists())
            self.assertEqual(app.capture_inputs(), default_case_inputs())
            app.close()
            restored = self.make_app(directory)
            self.assertEqual(restored.selected_case, 0)
            for case in restored.cases:
                self.assertEqual(case.draft, default_case_inputs())
                self.assertIsNone(case.result)
                self.assertIsNone(case.computed_inputs)
            self.assertEqual([c.color for c in restored.cases], ["#0072B2", "#D55E00", "#009E73"])
            self.assertTrue(all(v.get() for v in restored.case_visible))
            self.assertIsNone(restored.telemetry)
            self.assertTrue(restored.telemetry_visible.get())
            self.assertEqual(restored.telemetry_shift, 0)
            self.assertEqual(restored.telemetry_shift_input.get(), "0")
            self.assertEqual(restored.notebook.tab(restored.tabs["Booster"], "state"), "hidden")

    def test_missing_telemetry_keeps_settings_and_corrupt_state_uses_defaults(self):
        from TelemetryCSV import load_telemetry_csv
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            path = Path(directory) / "telemetry.csv"
            path.write_text("time_s,speed_mps\n0,10\n10,100\n")
            app.telemetry = load_telemetry_csv(path)
            app.payload_mass.set("13500")
            app.close()
            path.unlink()
            restored = self.make_app(directory)
            self.assertEqual(restored.payload_mass.get(), "13500")
            self.assertIsNone(restored.telemetry)
            self.assertIn("Telemetry could not be reopened", restored.status.get())
            restored.close()
            (Path(directory) / "gui_state.json").write_text('{broken')
            defaults = self.make_app(directory)
            self.assertEqual(defaults.capture_inputs(), default_case_inputs())
            self.assertIn("Could not restore", defaults.status.get())

    def test_autosave_and_close_during_recompute_preserve_previous_result(self):
        from GUIState import read_gui_state
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            case = app.cases[0]
            case.result = self.results["EXP"]
            case.request = self.requests["EXP"]
            case.computed_inputs = default_case_inputs()
            app.payload_mass.set("14000")
            self.assertIsNotNone(app.save_id)
            callback = app.root.tk.call("after", "info", app.save_id)[0]
            app.root.tk.call(callback)  # Execute the scheduled callback without sleeping.
            self.assertEqual(read_gui_state(app.state_path)["cases"][0]["draft"]["payload_mass"], "14000")
            case.state = "Computing (previous result retained)"
            app.process = Mock(poll=Mock(return_value=None))
            with patch.object(app, "cancel") as cancel:
                app.close()
            cancel.assert_called_once()
            restored = self.make_app(directory)
            self.assertEqual(restored.cases[0].state, "Interrupted; previous result retained")
            self.assertEqual(restored.payload_mass.get(), "14000")
            np.testing.assert_array_equal(restored.cases[0].result["x1"], case.result["x1"])
            self.assertIsNone(restored.process)

    def test_launch_sites_case_switch_persistence_custom_and_legacy_defaults(self):
        from LaunchSites import launch_site_inputs
        from GUIState import read_gui_state, write_gui_state
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            app.launch_site.set("Vandenberg")
            app.choose_launch_site()
            self.assertEqual(app.capture_inputs()["launch_site"], launch_site_inputs("Vandenberg"))
            self.assertEqual(app.orbit_variables["inclination"].get(), "28.6")
            app.select_case(1)
            app.launch_site.set("North Pole")
            app.choose_launch_site()
            self.assertEqual(app.site_variables["latitude"].get(), "90.0")
            app.site_variables["longitude"].set("45")
            self.assertEqual(app.launch_site.get(), "Custom")
            app.site_variables["altitude"].set("")
            app.close()
            restored = self.make_app(directory)
            self.assertEqual(restored.selected_case, 1)
            self.assertEqual(restored.launch_site.get(), "Custom")
            self.assertEqual(restored.site_variables["altitude"].get(), "")
            restored.select_case(0)
            self.assertEqual(restored.capture_inputs()["launch_site"], launch_site_inputs("Vandenberg"))
            restored.reset_gui()
            self.assertEqual(restored.capture_inputs()["launch_site"], launch_site_inputs())
            restored.close()
            path = Path(directory) / "gui_state.json"
            state = read_gui_state(path)
            for case in state["cases"]:
                case["draft"].pop("launch_site")
            state["cases"][0]["computed_inputs"] = deepcopy(state["cases"][0]["draft"])
            write_gui_state(path, state)
            legacy = self.make_app(directory)
            self.assertEqual(legacy.capture_inputs()["launch_site"], launch_site_inputs())
            self.assertFalse(legacy.cases[0].edited)

    def fit_bundle(self):
        request = nominal_request(asdict(DispesrionFactorsType()), payload_mass=12500)
        solution = deepcopy(self.results['EXP'])
        solution['payload_mass'] = 12500
        errors = {'stage1.speed': {'rmse': 20., 'unit': 'm/s', 'samples': 20, 'coverage': 1.}}
        baseline = dict(score=2., data_score=2., request=request, solution=solution, evaluation=1,
                        errors=errors, synchronization=dict(phase='stage1', shift=0.))
        best = deepcopy(baseline)
        best['request']['dispersion']['FirstStageThrust'] = 1.05
        best['solution']['dispersion']['FirstStageThrust'] = 1.05
        best['score'] = best['data_score'] = .5
        best['errors']['stage1.speed']['rmse'] = 10.
        best['synchronization']['shift'] = -2.5
        return dict(best=best, baseline=baseline, evaluations=3, termination='Evaluation budget reached',
                    parameters=['FirstStageThrust'], bounds={'FirstStageThrust': [.9, 1.1]},
                    history=[dict(evaluation=1, score=2., status='OK'), dict(evaluation=3, score=.5, status='OK')])

    def fit_gui_setup(self, directory):
        from TelemetryCSV import load_telemetry_csv
        app = self.make_app(directory)
        path = Path(directory) / 'telemetry.csv'
        path.write_text('time_s,speed_mps,altitude_m\n0,0,0\n5,10,10\n10,50,200\n20,100,1000\n')
        app.telemetry = load_telemetry_csv(path)
        app.refresh_comparison()
        options = dict(parameters=['FirstStageThrust'], bounds={'FirstStageThrust': [.9, 1.1]},
                       max_evaluations=8, altitude_scale=1000., speed_scale=25., regularization=.01)
        return app, options

    def test_fit_gui_loads_best_in_destination_syncs_and_persists_report(self):
        with tempfile.TemporaryDirectory() as directory:
            app, options = self.fit_gui_setup(directory)
            bundle = self.fit_bundle()
            app.cases[0].result = bundle['baseline']['solution']
            app.cases[0].request = bundle['baseline']['request']
            app.cases[0].computed_inputs = app.capture_inputs()
            original = app.cases[0].result
            self.assertEqual(str(app.fit_button['state']), 'normal')
            def start(command, **kwargs):
                self.assertTrue(command[1].endswith('DispersionFit.py'))
                folder = Path(command[-1])
                settings = json.loads((folder/'fit_inputs.json').read_text())
                self.assertEqual(settings['telemetry']['groups']['primary']['speed'], [0,10,50,100])
                with (folder/'fit_result.pickle').open('wb') as stream:
                    pickle.dump(bundle, stream)
                return Mock(poll=Mock(return_value=0))
            with patch('OptimizationGUI.subprocess.Popen', side_effect=start):
                app.start_fit(options, 1)
            self.assertIs(app.cases[0].result, original)
            self.assertEqual(app.selected_case, 1)
            self.assertEqual(float(app.variables['FirstStageThrust'].get()), 1.05)
            self.assertEqual(float(app.payload_mass.get()), 12500)
            self.assertEqual(app.telemetry_shift, -2.5)
            self.assertEqual(app.notebook.tab(app.notebook.select(), 'text'), 'Fit results')
            self.assertIn('Telemetry errors', app.fit_report_view.get('1.0','end'))
            self.assertEqual(len(app.figures['First stage'].axes[2].lines), 3)
            self.assertTrue((app.run_dir/'fit_report.json').exists())
            app.close()
            restored = self.make_app(directory)
            self.assertEqual(restored.cases[1].fit_report['best']['score'], .5)
            self.assertEqual(restored.notebook.tab(restored.notebook.select(), 'text'), 'Fit results')

    def test_cancelled_fit_keeps_checkpoint_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            app, options = self.fit_gui_setup(directory)
            process = Mock(poll=Mock(return_value=None))
            with patch('OptimizationGUI.subprocess.Popen', return_value=process):
                app.start_fit(options, 0)
            self.assertEqual(str(app.fit_button['state']), 'disabled')
            (app.run_dir/'fit_progress.json').write_text(json.dumps(dict(evaluation=4,max_evaluations=8,best_score=.5)))
            app.poll_fit_progress()
            self.assertIn('4/8', app.status.get())
            with (app.run_dir/'fit_best.pickle').open('wb') as stream:
                pickle.dump(self.fit_bundle(), stream)
            app.cancel()
            process.terminate.assert_called_once()
            app.root.after_cancel(app.poll_id)
            process.poll.return_value = -15
            app.poll()
            self.assertEqual(app.cases[0].fit_report['best']['score'], .5)
            self.assertIn('Cancelled', app.cases[0].fit_report['termination'])
            self.assertEqual(float(app.variables['FirstStageThrust'].get()), 1.05)

    def test_failed_fit_preserves_existing_case(self):
        with tempfile.TemporaryDirectory() as directory:
            app, options = self.fit_gui_setup(directory)
            original = self.fit_bundle()['baseline']
            case = app.cases[0]
            case.result, case.request = original['solution'], original['request']
            with patch('OptimizationGUI.subprocess.Popen', return_value=Mock(poll=Mock(return_value=1))):
                app.start_fit(options, 0)
            self.assertIs(case.result, original['solution'])
            self.assertIn('previous result retained', case.state)
            self.assertIsNone(case.fit_report)

    def fake_worker(self, result):
        def start(command, **kwargs):
            directory = Path(command[-1])
            filename = "solution.pickle" if result["success"] else "solution_Failed.pickle"
            with (directory / filename).open("wb") as stream:
                pickle.dump(result, stream)
            return Mock(poll=Mock(return_value=0))
        return start

    def test_gui_run_edits_dynamic_booster_tab_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            self.assertEqual(len(app.variables), len(asdict(DispesrionFactorsType())))
            self.assertEqual(app.notebook.tab(app.tabs["Booster"], "state"), "hidden")
            self.assertEqual(float(app.orbit_variables["apogee"].get()), 200.0)
            app.orbit_variables["apogee"].set("420")
            app.orbit_variables["perigee"].set("415")
            app.orbit_variables["inclination"].set("51.6")
            app.variables["FirstStageThrust"].set("1.02")
            app.recovery.set("ASDS — Drone ship")
            with patch("OptimizationGUI.subprocess.Popen", side_effect=self.fake_worker(self.results["ASDS"])):
                app.run()
            saved = json.loads((app.run_dir / "inputs.json").read_text())
            self.assertEqual(saved["dispersion"]["FirstStageThrust"], 1.02)
            self.assertEqual(saved["target_orbit"]["apogee"], 6798137.0)
            self.assertEqual(saved["target_orbit"]["perigee"], 6793137.0)
            self.assertAlmostEqual(saved["target_orbit"]["i"], np.deg2rad(51.6))
            self.assertIn("420 × 415 km · 51.6°", app.summary.get())
            target_lines = app.figures["Second stage"].axes[5].lines[-2:]
            np.testing.assert_allclose(target_lines[0].get_ydata(), [420, 420])
            np.testing.assert_allclose(target_lines[1].get_ydata(), [415, 415])
            self.assertEqual(app.notebook.tab(app.tabs["Booster"], "state"), "normal")
            self.assertIsNotNone(app.result)
            self.assertIn("Payload:", app.summary.get())
            expected_remaining = second_stage_propellant_remaining(app.cases[0].result)
            expected_label = f"S2 propellant left (est.): {expected_remaining:,.1f} kg"
            self.assertIn(expected_label, app.summary.get())
            self.assertIn(expected_label, app.case_status[0].get())
            app.variables["SecondStage_EmptyMass"].set("1000")
            self.assertIn(expected_label, app.case_status[0].get())
            app.recovery.set("EXP — Expendable")
            failed = {"success": False, "return_status": "Maximum_Iterations_Exceeded"}
            with patch("OptimizationGUI.subprocess.Popen", side_effect=self.fake_worker(failed)):
                app.run()
            self.assertIsNone(app.result)
            self.assertIn("did not converge", app.status.get())
            self.assertEqual(app.notebook.tab(app.tabs["Booster"], "state"), "normal")
            self.assertEqual(app.cases[0].request["recovery"], "ASDS")
            self.assertIn("previous result retained", app.case_status[0].get())
            self.assertEqual(str(app.run_button["state"]), "normal")
            app.reset_defaults()
            self.assertEqual(float(app.variables["FirstStageThrust"].get()), 1.0)

    def test_three_cases_overlay_colors_visibility_and_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            self.assertEqual(len(app.cases), 3)
            self.assertEqual(len({c.color for c in app.cases}), 3)
            for index, recovery in enumerate(("EXP", "ASDS", "RTLS")):
                app.select_case(index)
                self.assertEqual(app.payload_mass.get(), "-1")
                app.payload_mass.set(str(10000 + index * 1000))
                app.variables["FirstStageThrust"].set(str(1 + index / 100))
                app.recovery.set(next(label for label in (
                    "EXP — Expendable", "ASDS — Drone ship", "RTLS — Return to launch site")
                    if label.startswith(recovery)))
                result = dict(self.results[recovery], payload_mass=10000 + index * 1000)
                with patch("OptimizationGUI.subprocess.Popen", side_effect=self.fake_worker(result)):
                    app.run()
                self.assertIsNotNone(app.cases[index].result)
                self.assertEqual(app.cases[index].request["payload_mass_predefined"], 10000 + index * 1000)
                saved = json.loads((app.run_dir / "inputs.json").read_text())
                self.assertEqual(saved["payload_mass_predefined"], 10000 + index * 1000)
            first = app.figures["First stage"]
            self.assertEqual(len(first.axes), 7)  # Just one shared steering axis.
            self.assertEqual(len(first.axes[0].lines), 3)
            self.assertEqual(len(first.axes[6].lines), 3)
            self.assertEqual([line.get_color() for line in first.axes[0].lines],
                             [case.color for case in app.cases])
            self.assertEqual([line.get_label() for line in first.axes[0].lines],
                             [case.name for case in app.cases])
            self.assertEqual(len(app.figures["Booster"].axes[0].lines), 2)
            self.assertEqual(len(app.figures["Booster"].axes[6].lines), 2)
            self.assertEqual([curve.get_color() for curve in app.figures["Booster"].axes[6].lines],
                             [app.cases[1].color, app.cases[2].color])
            self.assertEqual(len(app.figures["Second stage"].axes[0].lines), 6)
            for tab, count in (("S1 details", 3), ("S2 propulsion", 6), ("S2 orbit", 6)):
                self.assertEqual(app.notebook.tab(app.tabs[tab], "state"), "normal")
                self.assertEqual(len(app.figures[tab].axes[0].lines), count)
                self.assertEqual(app.figures[tab].axes[0].lines[-1].get_color(), app.cases[2].color)

            app.case_visible[1].set(False)
            app.refresh_comparison()
            self.assertEqual(len(first.axes[0].lines), 2)
            self.assertEqual(len(app.figures["Booster"].axes[0].lines), 1)
            with patch("tkinter.colorchooser.askcolor", return_value=((128, 0, 128), "#800080")):
                app.choose_color(2)
            self.assertEqual(first.axes[0].lines[-1].get_color(), "#800080")
            self.assertEqual(app.figures["Booster"].axes[0].lines[0].get_color(), "#800080")
            app.case_visible[2].set(False)
            app.refresh_comparison()
            self.assertEqual(app.notebook.tab(app.tabs["Booster"], "state"), "hidden")
            app.case_visible[0].set(False)
            app.refresh_comparison()
            self.assertEqual(len(first.axes), 0)
            for tab in ("S1 details", "S2 propulsion", "S2 orbit"):
                self.assertEqual(len(app.figures[tab].axes), 0)
            self.assertIn("No cases displayed", app.summary.get())

            # Unsubmitted edits survive switching, including incomplete text.
            app.select_case(0)
            app.variables["FirstStageIsp"].set("")
            app.select_case(1)
            self.assertEqual(float(app.variables["FirstStageThrust"].get()), 1.01)
            self.assertEqual(app.payload_mass.get(), "11000")
            other_result = app.cases[1].result
            app.select_case(0)
            self.assertEqual(app.variables["FirstStageIsp"].get(), "")
            self.assertEqual(app.payload_mass.get(), "10000")
            self.assertTrue(app.cases[0].edited)
            self.assertEqual(app.cases[0].request["dispersion"]["FirstStageIsp"], 1.0)
            app.variables["FirstStageIsp"].set("1.05")
            app.case_visible[0].set(True)
            replacement = dict(self.results["EXP"], payload_mass=15000)
            with patch("OptimizationGUI.subprocess.Popen", side_effect=self.fake_worker(replacement)):
                app.run()
            self.assertEqual(app.cases[0].result["payload_mass"], 15000)
            self.assertEqual(app.cases[0].request["dispersion"]["FirstStageIsp"], 1.05)
            self.assertFalse(app.cases[0].edited)
            self.assertIs(app.cases[1].result, other_result)
            self.assertEqual(len(first.axes[0].lines), 1)
            self.assertEqual(app.run_button["text"], "Recompute Case 1")

    def test_cancelled_recompute_preserves_existing_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            with patch("OptimizationGUI.subprocess.Popen", side_effect=self.fake_worker(self.results["EXP"])):
                app.run()
            previous = app.cases[0].result
            app.orbit_variables["apogee"].set("500")
            process = Mock(poll=Mock(return_value=None))
            with patch("OptimizationGUI.subprocess.Popen", return_value=process):
                app.run()
            app.select_case(1)
            self.assertEqual(app.selected_case, 0)
            self.assertEqual(len(app.figures["First stage"].axes[0].lines), 1)
            app.cancel()
            app.root.after_cancel(app.poll_id)
            process.poll.return_value = -15
            app.poll()
            self.assertIs(app.cases[0].result, previous)
            self.assertIn("previous result retained", app.case_status[0].get())
            self.assertIn("200 × 200 km", app.summary.get())

    def test_load_telemetry_picker_overlay_frame_visibility_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flight.csv"
            path.write_text("time_s,altitude_km,speed_kmh,acceleration_g,altitude2_km,velocity2_kmh\n"
                            "10,1,360,2,100,7200\n20,3,720,3,110,7560\n")
            app = self.make_app(directory)
            result = deepcopy(self.results["EXP"])
            # The unsolved fixture can have a negative fairing-phase duration.
            # A successful worker result must have forward-running timestamps.
            for number in (2, 3):
                previous = result[f"t{number - 1}_vec"]
                count = np.asarray(result[f"t{number}_vec"]).size
                result[f"t{number}_vec"] = float(np.max(previous)) + np.arange(count) * 2.0
            with patch("OptimizationGUI.subprocess.Popen", side_effect=self.fake_worker(result)):
                app.run()
            saved_result = app.cases[0].result
            with patch("tkinter.filedialog.askopenfilename", return_value=str(path)) as picker:
                app.load_telemetry()
            picker.assert_called_once()
            self.assertEqual(app.telemetry.path, path)
            first = app.figures["First stage"].axes
            self.assertEqual(len(first[1].lines), 2)
            np.testing.assert_allclose(first[2].lines[-1].get_ydata(), [100, 200])
            self.assertEqual(len(first[5].lines), 2)
            self.assertEqual(first[5].lines[-1].get_label(), "Telemetry · derived")
            self.assertEqual(first[1].lines[-1].get_color(), app.telemetry_color)
            self.assertEqual(first[2].get_title(), "ECEF speed")
            np.testing.assert_allclose(first[2].lines[0].get_ydata(),
                                       np.linalg.norm(saved_result['x1'][2:4], axis=0))
            from TelemetryFrames import local_to_ecef_velocity
            state1 = saved_result['x1']
            ecef1 = local_to_ecef_velocity(np.vstack((state1[2], np.zeros_like(state1[2]), state1[3])),
                                           float(saved_result['LaunchAz']))
            np.testing.assert_allclose(app.figures['S1 details'].axes[3].lines[0].get_ydata(), ecef1[0])
            np.testing.assert_allclose(app.figures['S1 details'].axes[4].lines[0].get_ydata(), ecef1[2])
            self.assertEqual(app.figures['S1 details'].axes[3].get_title(), 'ECEF velocity X')
            self.assertNotIn(app.telemetry_color, [case.color for case in app.cases])
            acceleration_lines = {line.get_label(): line for line in app.figures["S1 details"].axes[1].lines}
            np.testing.assert_allclose(acceleration_lines["Telemetry · CSV"].get_ydata(), np.array([2, 3]) * 9.80665)
            self.assertNotIn("Telemetry · dv/dt (ECEF)", acceleration_lines)
            from TelemetryFrames import eci_to_ecef_velocity
            second_acceleration = app.figures["S2 propulsion"].axes[1]
            for curve, number in zip(second_acceleration.lines[:2], (2, 3)):
                t = np.asarray(saved_result[f"t{number}_vec"]).ravel()
                x = saved_result[f"x{number}"]
                speed = np.linalg.norm(eci_to_ecef_velocity(x[:3], x[3:6], t), axis=0)
                np.testing.assert_allclose(curve.get_ydata()[1:], np.diff(speed) / np.diff(t))
                self.assertIn("incl. gravity", curve.get_label())
            self.assertEqual(second_acceleration.get_title(), "Speed acceleration (ECEF, includes gravity)")
            np.testing.assert_allclose(app.figures["S2 propulsion"].axes[1].lines[-1].get_ydata(),
                                       [np.nan, 10], equal_nan=True)
            second = app.figures["Second stage"].axes[1]
            self.assertEqual(second.get_title(), "ECEF speed")
            state = saved_result["x2"]
            expected = np.linalg.norm(state[3:6] - np.cross([0, 0, 7.2921159e-5], state[:3].T).T, axis=0)
            np.testing.assert_allclose(second.lines[0].get_ydata(), expected)
            app.telemetry_shift_input.set("2.5")
            app.apply_telemetry_shift()
            np.testing.assert_allclose(app.figures["First stage"].axes[1].lines[-1].get_xdata(), [12.5, 22.5])
            app.telemetry_shift_input.set("nan")
            with patch("tkinter.messagebox.showerror") as error:
                app.apply_telemetry_shift()
            error.assert_called_once()
            self.assertEqual(app.telemetry_shift, 2.5)
            previous = app.telemetry
            path.write_text("time_s,unknown\n0,1\n")
            with patch("tkinter.filedialog.askopenfilename", return_value=str(path)), patch("tkinter.messagebox.showerror"):
                app.load_telemetry()
            self.assertIs(app.telemetry, previous)
            app.telemetry_visible.set(False)
            app.refresh_comparison()
            self.assertEqual(len(app.figures["First stage"].axes[1].lines), 1)
            self.assertEqual(app.figures["Second stage"].axes[1].get_title(), "Inertial velocity")
            self.assertEqual(app.figures['First stage'].axes[2].get_title(), 'Local velocity')
            self.assertEqual(app.figures['S1 details'].axes[3].get_title(), 'Local horizontal velocity')
            np.testing.assert_allclose(app.figures['S1 details'].axes[3].lines[0].get_ydata(), state1[2])
            np.testing.assert_allclose(app.figures['Second stage'].axes[1].lines[0].get_ydata(),
                                       np.linalg.norm(saved_result['x2'][3:6], axis=0))
            app.telemetry_visible.set(True)
            app.case_visible[0].set(False)
            app.refresh_comparison()
            self.assertEqual(len(app.figures["First stage"].axes[1].lines), 1)
            self.assertEqual(app.figures["First stage"].axes[1].lines[0].get_label(), "Telemetry")
            app.clear_telemetry()
            self.assertIsNone(app.telemetry)
            self.assertEqual(len(app.figures["First stage"].axes), 0)
            self.assertIs(app.cases[0].result, saved_result)

    def test_booster_heat_plot_respects_case_density_dispersion(self):
        nominal = self.results["ASDS"]
        denser = dict(nominal, dispersion=dict(nominal['dispersion'], AtmosphereDensity=4))
        figure = Figure()
        target = self.requests["ASDS"]["target_orbit"]
        draw_phase(figure, "Booster", nominal, target, color="blue", case_label="Nominal")
        draw_phase(figure, "Booster", denser, target, color="orange", case_label="Dense", append=True)
        self.assertEqual(len(figure.axes), 7)
        np.testing.assert_allclose(figure.axes[6].lines[1].get_ydata(),
                                   2 * figure.axes[6].lines[0].get_ydata())

    def test_sync_button_uses_selected_case_interpolation_and_absolute_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            self.assertEqual(str(app.sync_button['state']), 'disabled')
            for index, speed in enumerate((5, 15)):
                app.select_case(index)
                result = dict(self.results['EXP'])
                result['x1'] = result['x1'].copy()
                result['x1'][2:4, 0] = [speed, 0]
                with patch('OptimizationGUI.subprocess.Popen', side_effect=self.fake_worker(result)):
                    app.run()
            path = Path(directory) / 'sync.csv'
            path.write_text('time_s,speed_mps,altitude_km,velocity2_kmh,altitude2_km\n'
                            '10,0,0,1000,100\n14,10,1,1100,101\n18,20,2,1200,102\n')
            with patch('tkinter.filedialog.askopenfilename', return_value=str(path)):
                app.load_telemetry()
            app.select_case(0)
            self.assertEqual(app.sync_button['text'], 'Sync to Case 1')
            self.assertEqual(str(app.sync_button['state']), 'normal')
            initial_time = float(app.cases[0].result['t1_vec'][0])
            original_time = app.telemetry.time.copy()
            original_case_time = app.cases[0].result['t1_vec'].copy()
            app.telemetry_shift_input.set('99')
            app.apply_telemetry_shift()
            app.sync_telemetry()
            self.assertAlmostEqual(app.telemetry_shift, initial_time - 12)
            np.testing.assert_allclose(app.figures['First stage'].axes[2].lines[-1].get_xdata(), original_time + initial_time - 12)
            np.testing.assert_allclose(app.figures['Second stage'].axes[1].lines[-1].get_xdata(), original_time + initial_time - 12)
            self.assertIn('Synced to Case 1', app.status.get())
            app.sync_telemetry()
            self.assertAlmostEqual(app.telemetry_shift, initial_time - 12)
            np.testing.assert_array_equal(app.telemetry.time, original_time)
            np.testing.assert_array_equal(app.cases[0].result['t1_vec'], original_case_time)
            app.select_case(1)
            app.sync_telemetry()
            self.assertAlmostEqual(app.telemetry_shift, initial_time - 16)
            self.assertIn('Synced to Case 2', app.status.get())
            app.telemetry_primary.set('Stage 2')
            with patch('tkinter.messagebox.showerror') as error:
                app.sync_telemetry()
            error.assert_called_once()
            self.assertAlmostEqual(app.telemetry_shift, initial_time - 16)
            app.select_case(2)
            self.assertEqual(str(app.sync_button['state']), 'disabled')
            app.clear_telemetry()
            app.select_case(0)
            self.assertEqual(str(app.sync_button['state']), 'disabled')

    def test_gui_cancel_keeps_window_responsive_and_restores_run(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(directory)
            process = Mock(poll=Mock(return_value=None))
            with patch("OptimizationGUI.subprocess.Popen", return_value=process):
                app.run()
            self.assertEqual(str(app.run_button["state"]), "disabled")
            app.root.update()
            app.cancel()
            process.terminate.assert_called_once()
            app.root.after_cancel(app.poll_id)
            process.poll.return_value = -15
            app.poll()
            self.assertIsNone(app.process)
            self.assertEqual(str(app.run_button["state"]), "normal")
            self.assertIn("Cancelled", app.status.get())


if __name__ == "__main__":
    unittest.main()
