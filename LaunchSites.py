"""Approximate launch-site presets and validated geographic coordinates.

Kennedy retains the model's historical nominal coordinates. Real-site presets
are regional reference points, not surveyed pad positions; altitude defaults to
sea level. Equator and North Pole are idealized sites at zero longitude.
Sources: NASA/JPL Lunar Trajectories table 6-1; USACE Boca Chica public notice.
"""
import math

LAUNCH_SITES = {
    "Kennedy Space Center": (28.6, -80.6, 0.0),
    "Vandenberg": (34.77, -120.60, 0.0),
    "Starbase (Boca Chica)": (25.996, -97.154, 0.0),
    "Kourou": (5.24, -52.77, 0.0),
    "Equator": (0.0, 0.0, 0.0),
    "North Pole": (90.0, 0.0, 0.0),
    "Custom": None,
}
DEFAULT_LAUNCH_SITE = "Kennedy Space Center"


def launch_site_inputs(name=DEFAULT_LAUNCH_SITE):
    coordinates = LAUNCH_SITES[name]
    if coordinates is None:
        coordinates = LAUNCH_SITES[DEFAULT_LAUNCH_SITE]
    return dict(name=name, **dict(zip(("latitude", "longitude", "altitude"), map(str, coordinates))))


def parse_launch_site(values=None):
    values = launch_site_inputs() if values is None else values
    if not isinstance(values, dict) or values.get("name") not in LAUNCH_SITES:
        raise ValueError("Select a launch site or Custom coordinates.")
    site = {"name": values["name"]}
    for key in ("latitude", "longitude", "altitude"):
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Launch {key}: enter a number.") from None
        if not math.isfinite(value):
            raise ValueError(f"Launch {key}: enter a finite number.")
        site[key] = value
    if not -90 <= site["latitude"] <= 90:
        raise ValueError("Launch latitude must be between -90 and 90 degrees.")
    if not -180 <= site["longitude"] <= 180:
        raise ValueError("Launch longitude must be between -180 and 180 degrees (east positive).")
    if site["altitude"] <= -6378137:
        raise ValueError("Launch altitude must be above Earth's center.")
    return site
