import copy

from shared.flight_state import (
    make_flight_state, validate_flight_state,
    round_altitude, round_vertical_rate, round_track,
)

VALID = make_flight_state(
    timestamp=1790400021.0, icao24="a5d28c", flight_id="UAL3776 ",
    registration="N47412", type_code="B39M", category="A3",
    latitude=28.848358, longitude=-82.270907,
    altitude_baro_ft=14850, altitude_geom_ft=15575, on_ground=False,
    source="adsb_icao", accuracy_m=186, stale=False,
    ground_speed_kts=423.2, track_deg=174.17,
    vertical_rate_baro_fpm=-2560, vertical_rate_geom_fpm=None,
    selected_altitude_ft=12992,
    squawk="3456", emergency="none", seen_age_s=0.487, origin="live",
)

def test_builder():
    assert VALID["flight_id"] == "UAL3776", "trailing space should be stripped"
    assert validate_flight_state(VALID) == []

    minimal = make_flight_state(
        timestamp=0.0, icao24="aacf19",
        latitude=28.4, longitude=-81.3,
        on_ground=True, source="adsb_icao", origin="live",
    )
    assert minimal["flight_id"] == "aacf19", "no callsign falls back to icao24"
    assert minimal["status"]["squawk"] is None, "missing stays missing"
    assert validate_flight_state(minimal) == []

    blank = make_flight_state(
        timestamp=0.0, icao24="aacf19", flight_id="   ",
        latitude=28.4, longitude=-81.3,
        on_ground=True, source="adsb_icao", origin="live",
    )
    assert blank["flight_id"] == "aacf19", "blank callsign falls back to icao24"

INVALID_CASES = [
    ("missing section", lambda s: s.pop("nav")),
    ("extra key", lambda s: s["position"].update(foo=1)),
    ("missing field", lambda s: s["status"].pop("squawk")),
    ("section not a dict", lambda s: s.update(position=[])),
    ("ground as altitude", lambda s: s["position"].update(altitude_baro_ft="ground")),
    ("int as bool", lambda s: s["position"].update(on_ground=1)),
    ("bool as number", lambda s: s["kinematics"].update(ground_speed_kts=True)),
    ("non-octal squawk", lambda s: s["status"].update(squawk="7800")),
    ("short squawk", lambda s: s["status"].update(squawk="770")),
    ("squawk as emergency", lambda s: s["status"].update(emergency="7700")),
    ("unknown anomaly", lambda s: s.update(anomaly="hijack")),
    ("missing anomaly", lambda s: s.pop("anomaly")),
    ("unknown origin", lambda s: s.update(origin="fake")),
    ("unknown source", lambda s: s["position"].update(source="radar")),
    ("track 360", lambda s: s["kinematics"].update(track_deg=360)),
    ("latitude out of range", lambda s: s["position"].update(latitude=91)),
    ("negative speed", lambda s: s["kinematics"].update(ground_speed_kts=-1)),
    ("uppercase icao24", lambda s: s.update(icao24="A5D28C")),
    ("bad category", lambda s: s["aircraft"].update(category="A33")),
    ("required is None", lambda s: s.update(timestamp=None)),
]

def test_invalid():
    for description, change in INVALID_CASES:
        state = copy.deepcopy(VALID)
        change(state)
        assert validate_flight_state(state) != [], description
    assert validate_flight_state(None) != [], "not a dict"
    assert validate_flight_state(VALID) == [], "VALID was modified by a case"

def test_rounding():
    assert round_altitude(14862) == 14850
    assert round_vertical_rate(-2575) == -2560
    assert round_vertical_rate(-100) == -128
    assert round_track(359.996) == 0.0
    assert round_track(-10) == 350
    assert round_altitude(None) is None
    assert round_vertical_rate(None) is None
    assert round_track(None) is None

if __name__ == "__main__":
    test_builder()
    test_invalid()
    test_rounding()
    print("all tests passed")

    
