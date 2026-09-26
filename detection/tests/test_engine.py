from __future__ import annotations

import unittest

from detection.engine import DetectionEngine, build_default_engine
from detection.harness import make_state, smoke_scenarios
from detection.models import flight_state_from_mapping


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
        self.assertEqual(state.nav.selected_altitude_ft, 10_000.0)
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


if __name__ == "__main__":
    unittest.main()
