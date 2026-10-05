"""Portable custom vehicle definitions and the units used by their editor."""
import math


BUILTIN_VEHICLES = {"Falcon 9": 1, "Starship": 2, "Starship V3": 3}
TONNE_FORCE = 1000. * 9.80665
# Attribute, display label, display unit, conversion from display units to SI.
VEHICLE_PARAMETER_GROUPS = {
    "Geometry & mass": (
        ("Diameter", "Diameter", "m", 1.),
        ("EmptyFirstStageMass", "First-stage dry mass", "t", 1000.),
        ("EmptySecondStageMass", "Second-stage dry mass", "t", 1000.),
        ("FirstStagePropellentMass", "First-stage propellant mass", "t", 1000.),
        ("SecondStagePropellentMass", "Second-stage propellant mass", "t", 1000.),
        ("FairingMass", "Fairing mass", "t", 1000.),
    ),
    "Propulsion": (
        ("FirstStage_SL_Isp", "First-stage sea-level Isp", "s", 1.),
        ("FirstStage_Vac_Isp", "First-stage vacuum Isp", "s", 1.),
        ("FirstStage_SL_Thrust", "First-stage sea-level thrust", "tf", TONNE_FORCE),
        ("FirstStage_Vac_Thrust", "First-stage vacuum thrust", "tf", TONNE_FORCE),
        ("FirstStage_MinThrust_Factor", "First-stage minimum throttle", "fraction", 1.),
        ("FirstStage_MinThrust_IspFactor", "First-stage Isp at minimum throttle", "fraction", 1.),
        ("FirstStage_Ae", "First-stage total nozzle exit area", "m²", 1.),
        ("SecondStage_Thrust", "Second-stage thrust", "tf", TONNE_FORCE),
        ("SecondStage_Vac_Isp", "Second-stage vacuum Isp", "s", 1.),
        ("SecondStage_MinThrust_Factor", "Second-stage minimum throttle", "fraction", 1.),
        ("SecondStage_MinThrust_IspFactor", "Second-stage Isp at minimum throttle", "fraction", 1.),
    ),
    "Ascent aerodynamics": (
        ("FirstStage_CLa", "First-stage lift slope", "1/rad", 1.),
        ("FirstStage_Cd0", "First-stage base drag coefficient", "dimensionless", 1.),
        ("FirstStage_Cda2", "First-stage drag angle coefficient", "1/rad²", 1.),
        ("FirstStage_MaxAlpha", "Maximum angle of attack", "deg", math.pi / 180.),
        ("FairingSeparationAltitude", "Fairing separation altitude", "km", 1000.),
        ("FirstStage_MaxDynamicPressure", "Maximum ascent dynamic pressure", "kPa", 1000.),
        ("FirstStage_StageSeparationMaxDynamicPressure", "Maximum separation dynamic pressure", "kPa", 1000.),
        ("FirstStage_MaxQdynAlpha", "Maximum dynamic pressure × angle", "kPa·rad", 1000.),
    ),
    "Booster recovery": (
        ("Booster_Cd0", "Booster base drag coefficient", "dimensionless", 1.),
        ("Booster_Cda2", "Booster drag angle coefficient", "1/rad²", 1.),
        ("Booster_CLa", "Booster lift slope", "1/rad", 1.),
        ("Booster_MaxDynamicPressure", "Maximum return dynamic pressure", "kPa", 1000.),
        ("Booster_MaxHeatFlux", "Maximum return heat flux", "kW/m²", 1000.),
        ("Booster_k_empirical", "Empirical heat-flux coefficient", "model units", 1.),
    ),
    "Flight limits": (
        ("SecondStage_CoastTimeAfterSep", "Coast after stage separation", "s", 1.),
        ("Payload_Max_acc", "Maximum payload acceleration", "m/s²", 1.),
        ("ParkingOrbit_PerigeeAlt", "Minimum parking-orbit perigee", "km", 1000.),
    ),
}
VEHICLE_PARAMETERS = {item[0]: item for group in VEHICLE_PARAMETER_GROUPS.values() for item in group}
NONNEGATIVE_PARAMETERS = {
    "FairingMass", "FirstStage_Ae", "FirstStage_CLa", "FirstStage_Cd0", "FirstStage_Cda2",
    "FairingSeparationAltitude", "Booster_Cd0", "Booster_Cda2", "Booster_CLa",
    "Booster_k_empirical", "SecondStage_CoastTimeAfterSep",
}


def validate_vehicle_definition(definition):
    """Return a fresh definition in SI, rejecting missing or unsafe parameters."""
    if not isinstance(definition, dict) or set(definition) != {"name", "parameters"}:
        raise ValueError("A custom vehicle needs a name and its complete parameter set.")
    name = definition['name']
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError("Vehicle name must contain 1–80 characters.")
    name = name.strip()
    if any(ord(character) < 32 for character in name):
        raise ValueError("Vehicle name cannot contain control characters.")
    supplied = definition['parameters']
    if not isinstance(supplied, dict) or set(supplied) != set(VEHICLE_PARAMETERS):
        raise ValueError("Custom vehicle parameters must contain exactly all fields shown in the editor.")
    parameters = {}
    for key, (_, label, _, _) in VEHICLE_PARAMETERS.items():
        try:
            if isinstance(supplied[key], bool):
                raise ValueError
            value = float(supplied[key])
        except (TypeError, ValueError):
            raise ValueError(f"{label}: enter a number.") from None
        if not math.isfinite(value):
            raise ValueError(f"{label}: enter a finite number.")
        if value < 0 or value == 0 and key not in NONNEGATIVE_PARAMETERS:
            raise ValueError(f"{label}: must be {'nonnegative' if key in NONNEGATIVE_PARAMETERS else 'positive'}.")
        if key.endswith(('MinThrust_Factor', 'MinThrust_IspFactor')) and value > 1:
            raise ValueError(f"{label}: must be between 0 and 1.")
        if key == 'FirstStage_MaxAlpha' and value > math.pi / 2:
            raise ValueError("Maximum angle of attack cannot exceed 90 degrees.")
        parameters[key] = value
    return {'name': name, 'parameters': parameters}


def vehicle_editor_values(configuration):
    """Copy the selected nominal model into the editor's display units."""
    from LV_Type_scaled import VLType
    model = VLType()
    model.ApplyVehicleConfiguration(configuration)
    return {name: repr(float(getattr(model, name)) / spec[3]) for name, spec in VEHICLE_PARAMETERS.items()}


def vehicle_from_editor(name, values):
    if set(values) != set(VEHICLE_PARAMETERS):
        raise ValueError("Fill in every vehicle parameter.")
    parameters = {}
    for key, (_, label, _, scale) in VEHICLE_PARAMETERS.items():
        try:
            parameters[key] = float(values[key]) * scale
        except (TypeError, ValueError):
            raise ValueError(f"{label}: enter a number.") from None
    return validate_vehicle_definition({'name': name, 'parameters': parameters})


def vehicle_catalog(definitions=()):
    """Validate the saved library without changing built-in configurations."""
    if not isinstance(definitions, (list, tuple)):
        raise ValueError("Custom vehicles must be a list.")
    catalog = dict(BUILTIN_VEHICLES)
    names = {name.casefold() for name in catalog}
    for definition in definitions:
        definition = validate_vehicle_definition(definition)
        name = definition['name']
        if name.casefold() in names:
            raise ValueError(f"A vehicle named {name!r} already exists.")
        names.add(name.casefold())
        catalog[name] = definition
    return catalog


def vehicle_name(configuration):
    if isinstance(configuration, dict):
        return configuration['name']
    return next(name for name, number in BUILTIN_VEHICLES.items() if number == configuration)
