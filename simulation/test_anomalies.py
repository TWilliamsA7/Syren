from shared.flight_state import validate_flight_state
from simulation.anomalies import ANOMALIES, check_anomalies, ground_truth
from simulation.run import load_scenario, run_scenario

START = 1790400000
# MIA -> JAX at 36,000 ft: climbing until ~1000 s, cruising ~1000-1590 s, lands ~3035 s
MIA_TO_JAX = {
    "icao24": "~5f0001", "callsign": "SYN101", "type_code": "A320",
    "origin": "MIA", "destination": "JAX", "cruise_altitude_ft": 36000, "depart_at_s": 0,
}


def fly_with(anomaly_type, start_s, duration_s, params=None):
    """Fly MIA -> JAX with one anomaly; return every state, checked against the protocol."""
    scenario = {
        "start_time": START, "duration_s": 3600, "flights": [MIA_TO_JAX],
        "anomalies": [{"icao24": "~5f0001", "type": anomaly_type, "start_s": start_s,
                       "duration_s": duration_s, "params": params or {}}],
    }
    states = list(run_scenario(scenario))
    for state in states:
        assert validate_flight_state(state) == [], state
    return states


def at(states, t):
    """The fresh (not re-reported) state t seconds after the start, or None."""
    for state in states:
        if state["timestamp"] == START + t and state["status"]["seen_age_s"] == 0:
            return state
    return None


def angle_between(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_rapid_descent():
    states = fly_with("rapid_descent", 1100, 120, {"vertical_rate_fpm": -6000, "ramp_s": 60})
    assert -3500 < at(states, 1130)["kinematics"]["vertical_rate_baro_fpm"] < -2500
    assert at(states, 1200)["kinematics"]["vertical_rate_baro_fpm"] <= -5900
    assert at(states, 1200)["nav"]["selected_altitude_ft"] == 36000, "autopilot still at cruise"
    assert at(states, 1400)["kinematics"]["vertical_rate_baro_fpm"] > 0, "recovering"


def test_erratic_altitude():
    states = fly_with("erratic_altitude", 1100, 240, {"amplitude_fpm": 3000})
    window = [at(states, t) for t in range(1100, 1340)]
    rates = [s["kinematics"]["vertical_rate_baro_fpm"] for s in window]
    altitudes = [s["position"]["altitude_baro_ft"] for s in window]
    sign_changes = sum(1 for a, b in zip(rates, rates[1:]) if a * b < 0)
    assert sign_changes >= 3, rates
    assert max(altitudes) - min(altitudes) >= 800, (min(altitudes), max(altitudes))


def test_speed_loss():
    states = fly_with("speed_loss", 1100, 240, {"speed_kts": 200})
    assert at(states, 1300)["kinematics"]["ground_speed_kts"] == 200
    assert at(states, 1500)["kinematics"]["ground_speed_kts"] > 400, "recovers after"


def test_low_altitude_overspeed():
    states = fly_with("low_altitude_overspeed", 60, 300, {"speed_kts": 350, "altitude_ft": 2000})
    state = at(states, 300)
    assert state["position"]["altitude_baro_ft"] <= 2100, state["position"]
    assert state["kinematics"]["ground_speed_kts"] >= 340, state["kinematics"]


def test_route_deviation():
    states = fly_with("route_deviation", 1100, 300, {"offset_deg": 60})
    before = at(states, 1090)["kinematics"]["track_deg"]
    during = at(states, 1200)["kinematics"]["track_deg"]
    assert angle_between(before, during) >= 50, (before, during)


def test_squawk():
    states = fly_with("squawk", 1100, 300, {"code": "7700"})
    assert at(states, 1200)["status"]["squawk"] == "7700"
    assert at(states, 1200)["status"]["emergency"] == "general"
    assert at(states, 1500)["status"]["squawk"] == "1200", "back to normal after"
    assert at(states, 1500)["status"]["emergency"] == "none"


def test_signal_loss():
    states = fly_with("signal_loss", 1100, 180)
    assert all(at(states, t) is None for t in range(1100, 1280)), "no fresh positions"
    ages = [s["status"]["seen_age_s"] for s in states if s["status"]["seen_age_s"] > 0]
    assert max(ages) == 60, "re-reported for 60 s, then dropped"
    stale = {s["status"]["seen_age_s"]: s["position"]["stale"] for s in states}
    assert stale[20] is False and stale[21] is True
    assert at(states, 1280) is not None, "back after the window"


def test_engine_failure():
    states = fly_with("engine_failure", 1100, 1500, {"declare_after_s": 90})
    early = at(states, 1150)
    assert early["status"]["squawk"] != "7700", "not declared yet"
    assert early["kinematics"]["ground_speed_kts"] < 450
    assert early["kinematics"]["vertical_rate_baro_fpm"] < 0
    assert at(states, 1200)["status"]["squawk"] == "7700"
    assert at(states, 1500)["kinematics"]["ground_speed_kts"] == 260
    window = [at(states, t) for t in range(1100, 2000)]
    assert all(s["kinematics"]["vertical_rate_baro_fpm"] <= 0 for s in window), "never climbs"


def test_hijack():
    states = fly_with("hijack", 1100, 2600,
                      {"offset_deg": 120, "altitude_ft": 15000, "transponder_off_after_s": 240})
    assert at(states, 1150)["status"]["squawk"] == "7500"
    assert at(states, 1150)["status"]["emergency"] == "unlawful"
    before = at(states, 1090)["kinematics"]["track_deg"]
    assert angle_between(before, at(states, 1200)["kinematics"]["track_deg"]) >= 90
    assert all(at(states, t) is None for t in range(1340, 3600)), "transponder off"
    assert max(s["timestamp"] for s in states) < START + 1400, "gone from the feed"


def test_unknown_type_rejected():
    try:
        check_anomalies([{"type": "alien_abduction"}])
    except ValueError:
        return
    assert False, "unknown anomaly type should raise ValueError"


def test_ground_truth():
    scenario = {"start_time": 1000, "anomalies": [
        {"icao24": "~5f0001", "type": "speed_loss", "start_s": 100, "duration_s": 120}]}
    truth = ground_truth(scenario)
    assert truth[0]["start_time"] == 1100
    assert truth[0]["end_time"] == 1220
    assert truth[0]["icao24"] == "~5f0001"


def test_showcase_covers_every_anomaly():
    scenario = load_scenario("simulation/scenarios/anomaly_showcase.json")
    assert {a["type"] for a in scenario["anomalies"]} == set(ANOMALIES)
    for state in run_scenario(scenario):
        assert validate_flight_state(state) == [], state["icao24"]


if __name__ == "__main__":
    test_rapid_descent()
    test_erratic_altitude()
    test_speed_loss()
    test_low_altitude_overspeed()
    test_route_deviation()
    test_squawk()
    test_signal_loss()
    test_engine_failure()
    test_hijack()
    test_unknown_type_rejected()
    test_ground_truth()
    test_showcase_covers_every_anomaly()
    print("all tests passed")