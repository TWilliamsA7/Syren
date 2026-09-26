from __future__ import annotations

import unittest

from detection.engine import DetectionEngine, build_default_engine
from detection.harness import make_state, smoke_scenarios
from detection.models import flight_state_from_mapping
from simulation.run import run_scenario


START = 1_790_400_000
FLIGHT = {
    "icao24": "~5f0001", "callsign": "SYN101", "type_code": "A320",
    "origin": "MIA", "destination": "JAX", "cruise_altitude_ft": 36_000,
    "depart_at_s": 0,
}


def replay_simulation(anomaly: dict[str, object] | None = None):
    scenario = {
        "start_time": START, "duration_s": 3600, "flights": [FLIGHT],
        "anomalies": [anomaly] if anomaly else [], "noise_seed": 1,
    }
    engine = build_default_engine()
    for state in run_scenario(scenario):
        yield state, engine.update(state)


class DetectionEngineTests(unittest.TestCase):
    def test_generated_smoke_scenarios(self) -> None:
        for name, states, expected_type in smoke_scenarios():
            with self.subTest(scenario=name):
                engine = build_default_engine()
                results = [engine.update(state) for state in states]
                found = {anomaly.type for result in results for anomaly in result.anomalies}
                if expected_type is None:
                    self.assertFalse(found)
                else:
                    self.assertIn(expected_type, found)

    def test_fleet_conflict_is_reported_for_both_aircraft(self) -> None:
        engine = build_default_engine()
        first = make_state(timestamp=100, icao24="conflict1", longitude=0, track_deg=90, speed_kts=360)
        second = make_state(timestamp=100, icao24="conflict2", longitude=10 / 60, track_deg=270, speed_kts=360)
        results = engine.update_fleet((first, second))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(any(a.type == "AIRCRAFT_CONFLICT" for a in r.anomalies) for r in results))

    def test_updated_protocol_fields_and_result_mapping(self) -> None:
        state = flight_state_from_mapping(make_state(timestamp=1, icao24="ABC123"))
        self.assertEqual(state.icao24, "abc123")
        self.assertEqual(state.position.accuracy_m, 20.0)
        self.assertEqual(state.nav.selected_altitude_ft, 20_000.0)
        payload = build_default_engine().update(state).to_mapping()
        self.assertEqual(
            set(payload),
            {"icao24", "flight_id", "timestamp", "risk_score", "severity", "anomalies"},
        )

    def test_out_of_order_data_requires_reset(self) -> None:
        engine = DetectionEngine()
        engine.update(make_state(timestamp=100, icao24="order01"))
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            engine.update(make_state(timestamp=50, icao24="order01"))
        engine.reset("ORDER01")
        self.assertEqual(engine.update(make_state(timestamp=50, icao24="order01")).timestamp, 50)

    def test_duplicate_fleet_ids_are_rejected_before_processing(self) -> None:
        engine = DetectionEngine()
        repeated = make_state(timestamp=100, icao24="duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate icao24"):
            engine.update_fleet((repeated, repeated))
        self.assertEqual(engine.update(make_state(timestamp=50, icao24="duplicate")).timestamp, 50)

    def test_complete_nominal_flight_has_no_alerts(self) -> None:
        found = {anomaly.type for _, result in replay_simulation() for anomaly in result.anomalies}
        self.assertFalse(found, found)

    def test_generated_erratic_altitude_has_matching_alert(self) -> None:
        anomaly = {"icao24": FLIGHT["icao24"], "type": "erratic_altitude", "start_s": 1100,
                   "duration_s": 240, "params": {"amplitude_fpm": 3000}}
        found = {
            alert.type for state, result in replay_simulation(anomaly)
            if START + 1100 <= state["timestamp"] < START + 1340
            for alert in result.anomalies
        }
        self.assertIn("ALTITUDE_ANOMALY", found)

    def test_generated_speed_loss_has_matching_alert(self) -> None:
        anomaly = {"icao24": FLIGHT["icao24"], "type": "speed_loss", "start_s": 1100,
                   "duration_s": 240, "params": {"speed_kts": 200}}
        found = {
            alert.type for state, result in replay_simulation(anomaly)
            if START + 1100 <= state["timestamp"] < START + 1340
            for alert in result.anomalies
        }
        self.assertIn("SPEED_ANOMALY", found)

    def test_generated_low_altitude_overspeed_has_matching_alert(self) -> None:
        anomaly = {"icao24": FLIGHT["icao24"], "type": "low_altitude_overspeed", "start_s": 60,
                   "duration_s": 300, "params": {"speed_kts": 350, "altitude_ft": 2000}}
        found = {
            alert.type for state, result in replay_simulation(anomaly)
            if START + 60 <= state["timestamp"] < START + 360
            for alert in result.anomalies
        }
        self.assertIn("AGGRESSIVE_NEAR_GROUND_SPEED", found)

    def test_expected_descent_suppresses_contextual_rules_only(self) -> None:
        engine = build_default_engine()
        states = (
            make_state(timestamp=0, icao24="descent", altitude_ft=20_000, selected_altitude_ft=10_000, speed_kts=450, vertical_rate_fpm=-100),
            make_state(timestamp=30, icao24="descent", altitude_ft=19_900, selected_altitude_ft=10_000, speed_kts=420, vertical_rate_fpm=-300),
            make_state(timestamp=60, icao24="descent", altitude_ft=19_700, selected_altitude_ft=10_000, speed_kts=390, vertical_rate_fpm=-500),
        )
        found = {alert.type for state in states for alert in engine.update(state).anomalies}
        self.assertNotIn("SPEED_ANOMALY", found)
        self.assertNotIn("ACCELERATING_DESCENT", found)
        rapid = make_state(timestamp=61, icao24="descent", altitude_ft=19_600, selected_altitude_ft=10_000, vertical_rate_fpm=-4200)
        self.assertIn("RAPID_DESCENT", {alert.type for alert in engine.update(rapid).anomalies})

    def test_descent_speed_loss_below_normal_target_alerts(self) -> None:
        engine = build_default_engine()
        states = (
            make_state(timestamp=0, icao24="slowing", altitude_ft=17_000, selected_altitude_ft=3_000, speed_kts=350, vertical_rate_fpm=-2000),
            make_state(timestamp=30, icao24="slowing", altitude_ft=16_000, selected_altitude_ft=3_000, speed_kts=305, vertical_rate_fpm=-2000),
            make_state(timestamp=60, icao24="slowing", altitude_ft=15_000, selected_altitude_ft=3_000, speed_kts=260, vertical_rate_fpm=-2000),
        )
        found = {alert.type for state in states for alert in engine.update(state).anomalies}
        self.assertIn("SPEED_ANOMALY", found)

    def test_small_rate_noise_at_level_flight_is_not_accelerating_descent(self) -> None:
        engine = build_default_engine()
        states = (
            make_state(timestamp=0, icao24="level", altitude_ft=20_000, vertical_rate_fpm=64),
            make_state(timestamp=10, icao24="level", altitude_ft=20_000, vertical_rate_fpm=-64),
            make_state(timestamp=20, icao24="level", altitude_ft=20_000, vertical_rate_fpm=-128),
        )
        found = {alert.type for state in states for alert in engine.update(state).anomalies}
        self.assertNotIn("ACCELERATING_DESCENT", found)

    def test_normal_descent_rate_noise_does_not_alert(self) -> None:
        state = make_state(timestamp=0, icao24="normaldescent", altitude_ft=10_300,
                           selected_altitude_ft=3_000, vertical_rate_fpm=-3008)
        found = {alert.type for alert in build_default_engine().update(state).anomalies}
        self.assertNotIn("RAPID_DESCENT", found)


if __name__ == "__main__":
    unittest.main()
