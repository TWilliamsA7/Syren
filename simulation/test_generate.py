import gzip
import json
import os
import tempfile

from shared.flight_state import validate_flight_state
from simulation.anomalies import ANOMALIES
from simulation.flight_plan import AIRPORTS, distance_and_bearing
from simulation.generate import ELIGIBLE, flight_timeline, make_scenario, write_dataset
from simulation.run import run_scenario


def test_every_anomaly_type_has_a_rule():
    assert set(ELIGIBLE) == set(ANOMALIES)


def test_same_seed_same_scenario():
    assert make_scenario(1, 20, 0.5) == make_scenario(1, 20, 0.5)
    assert make_scenario(1, 20, 0.5) != make_scenario(2, 20, 0.5)


def test_flights_are_sensible():
    scenario = make_scenario(2, 50, 0.0)
    flights = scenario["flights"]
    assert scenario["anomalies"] == []
    assert len({f["icao24"] for f in flights}) == 50
    assert len({f["callsign"] for f in flights}) == 50
    for flight in flights:
        assert flight["origin"] != flight["destination"]
        assert flight["squawk"] not in ("1200", "7500", "7600", "7700")
        a, b = AIRPORTS[flight["origin"]], AIRPORTS[flight["destination"]]
        _, bearing = distance_and_bearing(a.latitude, a.longitude, b.latitude, b.longitude)
        odd = (flight["cruise_altitude_ft"] // 1000) % 2 == 1
        assert odd == (bearing < 180), (flight, bearing)


def test_anomalies_placed_in_flight():
    scenario = make_scenario(3, 100, 0.5)
    flights = {f["icao24"]: f for f in scenario["flights"]}
    anomalies = scenario["anomalies"]
    assert 30 <= len(anomalies) <= 70, len(anomalies)
    assert len({a["icao24"] for a in anomalies}) == len(anomalies), "at most one per flight"
    for anomaly in anomalies:
        flight = flights[anomaly["icao24"]]
        phase, altitude = flight_timeline(flight)[anomaly["start_s"] - flight["depart_at_s"]]
        assert ELIGIBLE[anomaly["type"]](phase, altitude), (anomaly, phase, altitude)


def test_dataset_files():
    scenario = make_scenario(4, 12, 0.5)
    with tempfile.TemporaryDirectory() as out_dir:
        counts = write_dataset(out_dir, scenario)
        with gzip.open(os.path.join(out_dir, "states.jsonl.gz"), "rt") as f:
            states = [json.loads(line) for line in f]
        with gzip.open(os.path.join(out_dir, "labels.jsonl.gz"), "rt") as f:
            labels = [json.loads(line) for line in f]
        with open(os.path.join(out_dir, "truth.json")) as f:
            windows = {entry["icao24"]: entry for entry in json.load(f)}
        assert os.path.exists(os.path.join(out_dir, "scenario.json"))

    assert len(states) == len(labels) == sum(counts.values())
    assert counts["normal"] > 0 and len(counts) > 1, counts
    for state, label in zip(states, labels):
        assert validate_flight_state(state) == [], state
        assert (label["icao24"], label["timestamp"]) == (state["icao24"], state["timestamp"])
        if label["label"] != "normal":
            window = windows[state["icao24"]]
            reported_at = state["timestamp"] + state["status"]["seen_age_s"]
            assert label["label"] == window["type"]
            assert window["start_time"] <= reported_at < window["end_time"]


def test_noise_is_repeatable():
    scenario = make_scenario(5, 3, 0.0)
    clean = {key: value for key, value in scenario.items() if key != "noise_seed"}
    noisy_run = list(run_scenario(scenario))
    assert noisy_run == list(run_scenario(scenario)), "same noise_seed, same values"
    assert noisy_run != list(run_scenario(clean)), "noise changes the reported values"


if __name__ == "__main__":
    test_every_anomaly_type_has_a_rule()
    test_same_seed_same_scenario()
    test_flights_are_sensible()
    test_anomalies_placed_in_flight()
    test_dataset_files()
    test_noise_is_repeatable()
    print("all tests passed")
