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


def run(ac, seconds):
    for _ in range(seconds):
        ac.step(1.0)


def test_speed_change():
    ac = make_aircraft(speed_kts=250, target_speed_kts=300)
    run(ac, 10)
    assert abs(ac.speed_kts - 265) < 1e-9, ac.speed_kts
    run(ac, 60)
    assert ac.speed_kts == 300, "should stop at the target, not overshoot"


def test_turn_shortest_way():
    right = make_aircraft(track_deg=350, target_track_deg=10)
    run(right, 1)
    assert abs(right.track_deg - 353) < 1e-9, right.track_deg
    run(right, 3)
    assert abs(right.track_deg - 2) < 1e-9, "should wrap through north"
    run(right, 10)
    assert abs(right.track_deg - 10) < 1e-9

    left = make_aircraft(track_deg=10, target_track_deg=350)
    run(left, 1)
    assert abs(left.track_deg - 7) < 1e-9, left.track_deg


def test_climb_levels_off():
    ac = make_aircraft(altitude_ft=3000, selected_altitude_ft=10000, climb_rate_fpm=2500)
    highest = 0
    for _ in range(600):
        ac.step(1.0)
        highest = max(highest, ac.altitude_ft)
    assert highest <= 10000 + 1, f"overshot to {highest}"
    assert abs(ac.altitude_ft - 10000) < 12.5, ac.altitude_ft
    assert abs(ac.vertical_rate_fpm) < 32, ac.vertical_rate_fpm


def test_target_vertical_rate_overrides_selected():
    ac = make_aircraft(selected_altitude_ft = 35000, target_vertical_rate_fpm=-6000)
    run(ac, 60)
    assert ac.vertical_rate_fpm == -6000
    assert ac.altitude_ft < 31000, ac.altitude_ft

if __name__ == "__main__":
    test_fly_west()
    test_fly_north()
    test_descent()
    test_flight_state()
    test_speed_change()
    test_turn_shortest_way()
    test_climb_levels_off()
    test_target_vertical_rate_overrides_selected()
    print("all tests passed")