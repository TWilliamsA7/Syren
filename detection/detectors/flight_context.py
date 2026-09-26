"""Small, source-independent context checks shared by temporal rules."""

from detection.models import FlightState


def expected_descent(state: FlightState) -> bool:
    """Recognize a commanded descent or low-altitude approach from observed data.

    Selected altitude is optional in the protocol. At low altitude, a negative
    vertical rate is enough to recognize the approach phase even when the
    selected altitude has already been crossed.
    """
    if state.position.on_ground is True or state.position.stale is True:
        return False
    altitude = state.position.altitude_baro_ft
    if altitude is None:
        altitude = state.position.altitude_geom_ft
    rate = state.kinematics.vertical_rate_baro_fpm
    if rate is None:
        rate = state.kinematics.vertical_rate_geom_fpm
    if altitude is None or rate is None or rate >= 0:
        return False
    selected = state.nav.selected_altitude_ft
    return altitude <= 10_000 or (selected is not None and selected < altitude)
