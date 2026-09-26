from shared.flight_state import validate_flight_state
from simulation.aircraft import SimAircraft


def make_aircraft(**overrides):
    fields = dict(
        icao24="~5f0001", callsign="SYN101", type_code="A320", category="A3",
        latitude=28.43, longitude=-81.31, altitude_ft=35000,
        speed_kts=460, track_deg=270,
    )
    fields.update(overrides)
    return SimAircraft(**fields)


def test_fly_west():
    ac = make_aircraft()
    for _ in range(60):
        ac.step(1.0)
    assert abs(ac.latitude - 28.43) < 1e-6, "flying west shouldn't change latitude"
    assert abs(ac.longitude - (-81.31 - 0.1453)) < 0.001, ac.longitude
    assert ac.altitude_ft == 35000


def test_fly_north():
    ac = make_aircraft(track_deg=0)
    for _ in range(60):
        ac.step(1.0)
    assert abs(ac.latitude - (28.43 + 460 / 60 / 60)) < 1e-6, ac.latitude
    assert abs(ac.longitude - (-81.31)) < 1e-9


def test_descent():
    ac = make_aircraft(vertical_rate_fpm=-2000)
    for _ in range(60):
        ac.step(1.0)
    assert abs(ac.altitude_ft - 33000) < 1e-6, ac.altitude_ft


def test_flight_state():
    ac = make_aircraft(vertical_rate_fpm=-2000, altitude_ft=33010)
    state = ac.to_flight_state(1790400021.0)
    assert validate_flight_state(state) == []
    assert state["origin"] == "sim"
    assert state["position"]["altitude_baro_ft"] == 33000
    assert state["kinematics"]["vertical_rate_baro_fpm"] == -1984


if __name__ == "__main__":
    test_fly_west()
    test_fly_north()
    test_descent()
    test_flight_state()
    print("all tests passed")