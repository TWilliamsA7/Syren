from shared.flight_state import validate_flight_state
from simulation.run import load_scenario, run_scenario

SCENARIO_PATH = "simulation/scenarios/normal_traffic.json"


def two_flights(duration_s):
    return {
        "start_time": 1790400000,
        "duration_s": duration_s,
        "flights": [
            {"icao24": "~5f0001", "callsign": "SYN101", "type_code": "A320",
             "origin": "MCO", "destination": "TPA", "cruise_altitude_ft": 16000,
             "depart_at_s": 0},
            {"icao24": "~5f0002", "callsign": "SYN202", "type_code": "B738",
             "origin": "TPA", "destination": "MCO", "cruise_altitude_ft": 15000,
             "depart_at_s": 120},
        ],
    }


def test_scenario_file():
    scenario = load_scenario(SCENARIO_PATH)
    states = list(run_scenario(scenario))
    assert states, "scenario produced nothing"

    timestamps = [s["timestamp"] for s in states]
    assert timestamps == sorted(timestamps), "states must be in time order"

    for state in states:
        assert validate_flight_state(state) == [], state["icao24"]

    first_seen = {}
    for state in states:
        first_seen.setdefault(state["icao24"], state["timestamp"])
    for flight in scenario["flights"]:
        expected = scenario["start_time"] + flight["depart_at_s"]
        assert first_seen[flight["icao24"]] == expected, flight["icao24"]


def test_departure_time():
    states = list(run_scenario(two_flights(60)))
    assert {s["icao24"] for s in states} == {"~5f0001"}, "second flight departs at 120 s"


def test_flights_land_and_leave():
    states = list(run_scenario(two_flights(3 * 3600)))
    last = {}
    for state in states:
        last[state["icao24"]] = state
    for icao24, state in last.items():
        assert state["position"]["on_ground"], icao24
        assert state["kinematics"]["ground_speed_kts"] == 0, icao24
    assert states[-1]["timestamp"] < 1790400000 + 3 * 3600, "landed aircraft should be removed"


if __name__ == "__main__":
    test_scenario_file()
    test_departure_time()
    test_flights_land_and_leave()
    print("all tests passed")