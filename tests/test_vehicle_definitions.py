"""Custom vehicles survive editor, worker, model, and GUI-state boundaries."""
from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import casadi as ca
import numpy as np

from DispersionFit import default_fit_bounds, _solve_candidate
from GUIState import read_gui_state, write_gui_state
from LV_Optimization_Type import LV_Optimization
from LV_Type_scaled import DispesrionFactorsType, VLType
from OptimizationGUI import (ComparisonCase, OptimizationApp, nominal_dispersion_values,
                             nominal_request, run_optimization, worker)
from VehicleDefinitions import (BUILTIN_VEHICLES, VEHICLE_PARAMETERS, vehicle_catalog,
                                vehicle_editor_values, vehicle_from_editor, vehicle_name,
                                validate_vehicle_definition, read_vehicle_definition,
                                write_vehicle_definition)


class VehicleDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.values = vehicle_editor_values(1)
        self.definition = vehicle_from_editor('Research vehicle', self.values)

    def test_editor_covers_all_independent_vehicle_parameters(self):
        common_or_derived = {'LaunchLatitude', 'LaunchLongitude', 'LaunchAltitude', 'g0',
                             'R0', 'e', 'omega_earth', 'mu', 'Sref', 'LV_total_mass',
                             'FirstStage_EmptyMass', 'SecondStage_EmptyMass',
                             'SecondStage_FullMass', 'TotalPropellentMass'}
        physical = {field.name for field in fields(VLType) if field.type is float}
        self.assertEqual(set(VEHICLE_PARAMETERS), physical - common_or_derived)

    def test_each_selected_vehicle_can_be_cloned_without_changing_its_parameters(self):
        for configuration in BUILTIN_VEHICLES.values():
            with self.subTest(configuration=configuration):
                expected = VLType()
                expected.ApplyVehicleConfiguration(configuration)
                definition = vehicle_from_editor('Clone', vehicle_editor_values(configuration))
                model = VLType()
                model.ApplyVehicleConfiguration(definition)
                for name in VEHICLE_PARAMETERS:
                    self.assertAlmostEqual(getattr(model, name), getattr(expected, name))
                cloned = vehicle_from_editor('Clone of clone', vehicle_editor_values(definition))
                for name, value in definition['parameters'].items():
                    self.assertAlmostEqual(cloned['parameters'][name], value)
                self.assertAlmostEqual(model.Sref, np.pi * (expected.Diameter / 2)**2)

    def test_new_glenn_selector_reaches_worker_and_all_vehicle_phases(self):
        configuration = vehicle_catalog()['New Glenn']
        self.assertEqual(configuration, 4)
        self.assertEqual(vehicle_name(configuration), 'New Glenn')
        request = nominal_request(asdict(DispesrionFactorsType()), configuration, 'ASDS')
        model = LV_Optimization(DispesrionFactorsType(), request['configuration'])
        for phase in (model.booster, model.eci, model.boostback, model.rocket_return):
            self.assertEqual(phase.Diameter, 7.)
            self.assertEqual(phase.FirstStage_SL_Thrust, 7*2846e3)
            self.assertEqual(phase.SecondStage_Thrust, 2*890e3)
            self.assertEqual(phase.SecondStage_Vac_Isp, 445.)
            self.assertEqual(phase.LV_total_mass, 1547e3)
            self.assertAlmostEqual(phase.Sref, np.pi*3.5**2)
        with patch('LV_Optimization_Type.LV_Optimization') as constructor:
            run_optimization(request, None)
        self.assertEqual(constructor.call_args.args[1], configuration)
        self.assertEqual(constructor.return_value.SolveOptimiztion.call_args.kwargs['RecoveryStrategy'], 'ASDS')

    def test_display_units_convert_to_si_and_nominal_annotations(self):
        values = dict(self.values, EmptyFirstStageMass='30', FirstStage_SL_Thrust='1000',
                      FirstStage_SL_Isp='290', FirstStage_MaxAlpha='12')
        definition = vehicle_from_editor('Edited', values)
        self.assertEqual(definition['parameters']['EmptyFirstStageMass'], 30000)
        self.assertEqual(definition['parameters']['FirstStage_SL_Thrust'], 9806650)
        self.assertEqual(definition['parameters']['FirstStage_SL_Isp'], 290)
        self.assertAlmostEqual(definition['parameters']['FirstStage_MaxAlpha'], np.deg2rad(12))
        labels = nominal_dispersion_values(definition)
        self.assertEqual(labels['FirstStage_EmptyMass'], '30 t')
        self.assertIn('SL 1,000', labels['FirstStageThrust'])
        self.assertIn('SL 290', labels['FirstStageIsp'])
        bounds = default_fit_bounds({'configuration': definition})
        original = default_fit_bounds({'configuration': 1})
        self.assertAlmostEqual(bounds['FirstStage_EmptyMass'][1] / original['FirstStage_EmptyMass'][1], 30 / 25.6)

    def test_invalid_definitions_and_duplicate_names_are_rejected(self):
        for key, value in (('Diameter', 0), ('EmptyFirstStageMass', -1),
                           ('SecondStage_Thrust', 'NaN'), ('FirstStage_MinThrust_Factor', 1.01),
                           ('SecondStage_MinThrust_IspFactor', 0), ('FirstStage_MaxAlpha', 91),
                           ('Payload_Max_acc', 'inf')):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                vehicle_from_editor('Bad', dict(self.values, **{key: value}))
        for name in ('', '  ', 'x' * 81, 'bad\nname'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                vehicle_from_editor(name, self.values)
        for parameters in ({}, dict(self.definition['parameters'], nx=5),
                           dict(self.definition['parameters'], Diameter=True)):
            with self.assertRaises(ValueError):
                validate_vehicle_definition(dict(self.definition, parameters=parameters))
        for name in ('Falcon 9', 'falcon 9', 'Research vehicle', 'research vehicle'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                vehicle_catalog([self.definition, vehicle_from_editor(name, self.values)])

    def test_definition_reaches_every_phase_and_dispersion_is_applied_once(self):
        definition = vehicle_from_editor('Edited', dict(self.values, Diameter='4',
                    EmptyFirstStageMass='30', FirstStage_SL_Isp='290', Booster_Cd0='1.7'))
        dispersion = DispesrionFactorsType(FirstStage_EmptyMass=500, FirstStageIsp=1.02,
                                           FirstStageThrust=1.03, BoosterStageCx0=1.1)
        model = LV_Optimization(dispersion, definition)
        for phase in (model.booster, model.eci, model.boostback, model.rocket_return):
            self.assertAlmostEqual(phase.EmptyFirstStageMass, 30500)
            self.assertAlmostEqual(phase.FirstStage_SL_Isp, 290 * 1.02)
            self.assertAlmostEqual(phase.FirstStage_SL_Thrust, definition['parameters']['FirstStage_SL_Thrust'] * 1.03)
            self.assertAlmostEqual(phase.Booster_Cd0, 1.7 * 1.1)
            self.assertAlmostEqual(phase.Sref, np.pi * 4)
        self.assertEqual(definition['parameters']['EmptyFirstStageMass'], 30000)

    def test_optimization_worker_and_fit_receive_self_contained_definitions(self):
        request = nominal_request(asdict(DispesrionFactorsType()), configuration=self.definition)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'inputs.json').write_text(json.dumps(request))
            with patch('LV_Optimization_Type.LV_Optimization') as constructor:
                constructor.return_value.SolveOptimiztion.return_value = {'success': True}
                worker(path)
            self.assertEqual(constructor.call_args.args[1], self.definition)
            self.assertTrue((path / 'solution.pickle').exists())
        with patch('LV_Optimization_Type.LV_Optimization') as constructor:
            _solve_candidate(json.loads(json.dumps(request)))
        self.assertEqual(constructor.call_args.args[1], self.definition)
        self.definition['parameters']['Diameter'] = 5
        self.assertEqual(request['configuration']['parameters']['Diameter'], 3.66)

    def test_custom_solution_builds_all_recovery_phases(self):
        class InitialValues:
            def __init__(self, opti):
                self.opti = opti
            def value(self, expression):
                return self.opti.debug.value(expression, self.opti.initial())
        for recovery in ('EXP', 'ASDS', 'RTLS'):
            with self.subTest(recovery=recovery):
                request = nominal_request(asdict(DispesrionFactorsType()), self.definition, recovery)
                with patch.object(ca.Opti, 'solve', lambda opti: InitialValues(opti)), \
                     patch.object(ca.Opti, 'return_status', return_value='test'), \
                     patch.object(ca.Opti, 'stats', return_value={'success': True, 'iter_count': 0}):
                    result = run_optimization(request, None)
                self.assertEqual(result['configuration'], self.definition)
                self.assertEqual(vehicle_name(result['configuration']), 'Research vehicle')
                self.assertTrue(np.isfinite(result['x1']).all())
                self.assertEqual('x4' in result, recovery != 'EXP')

    def test_new_vehicle_registration_preserves_builtins_and_other_cases(self):
        app = OptimizationApp.__new__(OptimizationApp)
        app.vehicles = vehicle_catalog()
        app.vehicle_combo = Mock()
        app.vehicle = Mock()
        app.status = Mock()
        app.schedule_save = Mock()
        app.cases = [ComparisonCase('Case 1', 'blue'), ComparisonCase('Case 2', 'red')]
        other = deepcopy(app.cases[1].draft)
        app.add_vehicle(self.definition)
        self.assertEqual(app.vehicles['Falcon 9'], 1)
        self.assertEqual(app.vehicles['Research vehicle'], self.definition)
        app.vehicle.set.assert_called_once_with('Research vehicle')
        self.assertEqual(app.cases[1].draft, other)
        app.schedule_save.assert_called_once()
        before = deepcopy(app.vehicles)
        with self.assertRaises(ValueError):
            app.add_vehicle(self.definition)
        self.assertEqual(app.vehicles, before)

    def test_vehicle_library_and_requests_round_trip_with_gui_state(self):
        request = nominal_request(asdict(DispesrionFactorsType()), self.definition)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gui_state.json'
            state = {'cases': [{'draft': {'vehicle': 'Research vehicle'}, 'request': request}, {}, {}],
                     'custom_vehicles': [self.definition]}
            write_gui_state(path, state)
            restored = read_gui_state(path)
        catalog = vehicle_catalog(restored['custom_vehicles'])
        self.assertEqual(catalog['Research vehicle'], restored['cases'][0]['request']['configuration'])
        self.assertEqual(vehicle_catalog(), BUILTIN_VEHICLES)

    def test_standalone_vehicle_round_trip_and_failed_save_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rocket.json'
            write_vehicle_definition(path, self.definition)
            self.assertEqual(read_vehicle_definition(path), self.definition)
            before = path.read_bytes()
            changed = vehicle_from_editor('Updated', dict(self.values, Diameter='5'))
            with patch('VehicleDefinitions.os.replace', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    write_vehicle_definition(path, changed)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])
            with self.assertRaises(ValueError):
                write_vehicle_definition(path, dict(self.definition, parameters={}))
            self.assertEqual(path.read_bytes(), before)

    def test_standalone_vehicle_rejects_invalid_formats_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rocket.json'
            for payload in ('{broken', '[]', '{}',
                            json.dumps({'format': 'launch-vehicle-configuration', 'version': 999}),
                            json.dumps({'format': 'launch-vehicle-configuration', 'version': True,
                                        'vehicle': self.definition}),
                            json.dumps({'format': 'launch-vehicle-configuration', 'version': 1,
                                        'vehicle': dict(self.definition, parameters={})})):
                with self.subTest(payload=payload):
                    path.write_text(payload)
                    with self.assertRaises(ValueError):
                        read_vehicle_definition(path)

    def library_app(self):
        app = OptimizationApp.__new__(OptimizationApp)
        app.process = None
        app.root = Mock()
        app.output_root = Path('/tmp')
        app.vehicles = vehicle_catalog([self.definition])
        app.vehicle_combo = Mock()
        app.vehicle = Mock(get=Mock(return_value=self.definition['name']))
        app.status = Mock()
        app.schedule_save = Mock()
        app.update_case_labels = Mock()
        app.cases = [ComparisonCase(f'Case {i + 1}', 'blue') for i in range(3)]
        return app

    def test_edit_and_rename_update_referencing_drafts_preserving_computed_results(self):
        app = self.library_app()
        for case in app.cases[:2]:
            case.draft['vehicle'] = self.definition['name']
            case.draft['vehicle_definition'] = deepcopy(self.definition)
            case.computed_inputs = deepcopy(case.draft)
            case.request = nominal_request(asdict(DispesrionFactorsType()), self.definition)
            case.result = {'payload_mass': 10000}
        computed = deepcopy(app.cases[0].computed_inputs)
        request = deepcopy(app.cases[0].request)
        result = app.cases[0].result
        unrelated = deepcopy(app.cases[2].draft)
        changed = vehicle_from_editor(self.definition['name'], dict(self.values, Diameter='5'))
        app.update_vehicle(self.definition['name'], changed)
        for case in app.cases[:2]:
            self.assertTrue(case.edited)
            self.assertEqual(case.draft['vehicle_definition'], changed)
        self.assertEqual(app.cases[0].computed_inputs, computed)
        self.assertEqual(app.cases[0].request, request)
        self.assertIs(app.cases[0].result, result)
        renamed = dict(changed, name='Renamed rocket')
        app.update_vehicle(self.definition['name'], renamed)
        self.assertNotIn(self.definition['name'], app.vehicles)
        for case in app.cases[:2]:
            self.assertEqual(case.draft['vehicle'], 'Renamed rocket')
        app.vehicle.set.assert_called_with('Renamed rocket')
        self.assertEqual(app.cases[2].draft, unrelated)
        before = deepcopy(app.vehicles)
        with self.assertRaises(ValueError):
            app.update_vehicle('Renamed rocket', dict(renamed, name='Falcon 9'))
        self.assertEqual(app.vehicles, before)
        with self.assertRaises(ValueError):
            app.update_vehicle('Falcon 9', renamed)

    def test_import_reuses_identical_definition_and_preserves_name_conflicts(self):
        app = self.library_app()
        before = deepcopy(app.vehicles)
        self.assertEqual(app.import_vehicle(self.definition), self.definition['name'])
        self.assertEqual(app.vehicles, before)
        changed = vehicle_from_editor(self.definition['name'], dict(self.values, Diameter='5'))
        name = app.import_vehicle(changed)
        self.assertEqual(name, 'Research vehicle (2)')
        self.assertEqual(app.vehicles[self.definition['name']], self.definition)
        self.assertEqual(app.vehicles[name]['parameters']['Diameter'], 5)
        self.assertEqual(changed['name'], self.definition['name'])
        builtin = vehicle_from_editor('Falcon 9', self.values)
        self.assertEqual(app.import_vehicle(builtin), 'Falcon 9 (2)')
        self.assertEqual(app.vehicles['Falcon 9'], 1)
        long_name = vehicle_from_editor('x' * 80, self.values)
        app.import_vehicle(long_name)
        self.assertEqual(len(app.import_vehicle(dict(long_name, parameters=changed['parameters']))), 80)

    def test_save_and_load_controls_export_definition_and_handle_invalid_import(self):
        app = self.library_app()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rocket.json'
            with patch('tkinter.filedialog.asksaveasfilename', return_value=str(path)):
                app.save_vehicle_file()
            self.assertEqual(read_vehicle_definition(path), self.definition)
            app.vehicles = vehicle_catalog()
            with patch('tkinter.filedialog.askopenfilename', return_value=str(path)):
                app.load_vehicle_file()
            self.assertEqual(app.vehicles[self.definition['name']], self.definition)
            app.vehicle.set.assert_called_with(self.definition['name'])
            before = deepcopy(app.vehicles)
            path.write_text('{broken')
            with patch('tkinter.filedialog.askopenfilename', return_value=str(path)), \
                 patch('tkinter.messagebox.showerror') as showerror:
                app.load_vehicle_file()
            showerror.assert_called_once()
            self.assertEqual(app.vehicles, before)
            app.vehicle.get.return_value = 'Falcon 9'
            with patch('tkinter.filedialog.asksaveasfilename', return_value=str(path)):
                app.save_vehicle_file()
            self.assertEqual(read_vehicle_definition(path)['parameters'],
                             vehicle_from_editor('Falcon 9', self.values)['parameters'])

    def test_vehicle_file_controls_cancel_and_are_guarded_during_runs(self):
        app = self.library_app()
        with patch('tkinter.filedialog.asksaveasfilename', return_value='') as save, \
             patch('tkinter.filedialog.askopenfilename', return_value='') as load, \
             patch('OptimizationGUI.write_vehicle_definition') as write:
            app.save_vehicle_file()
            app.load_vehicle_file()
            write.assert_not_called()
            save.reset_mock()
            load.reset_mock()
            app.process = Mock()
            app.save_vehicle_file()
            app.load_vehicle_file()
            app.open_vehicle_dialog(edit=True)
            save.assert_not_called()
            load.assert_not_called()

    def test_edit_dialog_applies_changes_to_selected_custom_vehicle(self):
        app = self.library_app()
        app.update_vehicle = Mock()
        app.add_vehicle = Mock()
        widgets = {name: Mock() for name in ('Frame', 'Label', 'Entry', 'Notebook', 'Scrollbar', 'Button')}
        def variable(**kwargs):
            return Mock(get=Mock(return_value=kwargs['value']))
        with patch.multiple('tkinter', Toplevel=Mock(), Canvas=Mock(), StringVar=variable), \
             patch.multiple('tkinter.ttk', **widgets):
            app.open_vehicle_dialog(edit=True)
        buttons = {call.kwargs['text']: call.kwargs['command'] for call in widgets['Button'].call_args_list}
        buttons['Apply changes']()
        selected, definition = app.update_vehicle.call_args.args
        self.assertEqual(selected, self.definition['name'])
        self.assertEqual(definition, self.definition)
        app.add_vehicle.assert_not_called()

    def test_dialog_starts_from_current_vehicle_and_creates_validated_copy(self):
        app = OptimizationApp.__new__(OptimizationApp)
        app.process = None
        app.root = Mock()
        app.vehicles = vehicle_catalog()
        app.vehicle = Mock(get=Mock(return_value='Starship V3'))
        app.add_vehicle = Mock()
        variables = []
        def variable(**kwargs):
            value = Mock(get=Mock(return_value=kwargs['value']))
            variables.append(value)
            return value
        widgets = {name: Mock() for name in ('Frame', 'Label', 'Entry', 'Notebook', 'Scrollbar', 'Button')}
        with patch.multiple('tkinter', Toplevel=Mock(), Canvas=Mock(), StringVar=variable), \
             patch.multiple('tkinter.ttk', **widgets):
            app.open_vehicle_dialog()
        self.assertEqual(variables[0].get(), 'Starship V3 copy')
        expected = vehicle_editor_values(3)
        self.assertEqual([variable.get() for variable in variables[1:]], list(expected.values()))
        buttons = {call.kwargs['text']: call.kwargs['command'] for call in widgets['Button'].call_args_list}
        buttons['Create vehicle']()
        definition = app.add_vehicle.call_args.args[0]
        self.assertEqual(definition['name'], 'Starship V3 copy')
        self.assertEqual(definition['parameters']['FirstStagePropellentMass'], 3650000)


if __name__ == '__main__':
    unittest.main()
