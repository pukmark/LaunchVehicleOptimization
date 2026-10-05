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
                                validate_vehicle_definition)


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
