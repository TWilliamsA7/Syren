"""Focused checks for the per-aircraft predictive warning boundary."""

from __future__ import annotations

import unittest

from detection.models import FlightState, Kinematics, Position, Status
from ml.aircraft_warning import AircraftWarningEngine
from ml.replay_aircraft_warnings import summarize_mode


def state(icao: str, time: float, speed: float, altitude: float = 10000.0,
          flight_id: str = "FLT1", on_ground: bool = False, squawk: str | None = None) -> FlightState:
    return FlightState(
        timestamp=time, icao24=icao, flight_id=flight_id,
        position=Position(altitude_baro_ft=altitude, on_ground=on_ground),
        kinematics=Kinematics(ground_speed_kts=speed),
        status=Status(squawk=squawk),
    )


class AircraftWarningTests(unittest.TestCase):
    def test_speed_loss_is_causal_and_aircraft_isolated(self) -> None:
        engine = AircraftWarningEngine()
        results_a = []
        results_b = []
        for index in range(11):
            now = index * 30.0
            results_a.append(engine.update(state("aaaaaa", now, 200 - 5 * index)))
            results_b.append(engine.update(state("bbbbbb", now, 200, squawk="7700")))
        self.assertFalse(any(result.alert for result in results_a[:9]))
        self.assertTrue(any(result.alert for result in results_a[9:]))
        self.assertFalse(any(result.alert for result in results_b))
        warning = next(result for result in results_a if result.alert)
        self.assertEqual([signal.type for signal in warning.signals], ["SPEED_LOSS"])
        self.assertEqual(warning.to_mapping()["status"], "warning")

    def test_altitude_rule_and_suppression(self) -> None:
        engine = AircraftWarningEngine("altitude_reversal")
        altitudes = [10000, 10000, 10100, 10200, 10300, 10400,
                     10500, 10400, 10300, 10200, 10100, 10000, 10000]
        results = [engine.update(state("aaaaaa", index * 30.0, 150, altitude))
                   for index, altitude in enumerate(altitudes)]
        self.assertEqual(sum(result.alert for result in results), 1)
        self.assertTrue(any(result.status == "suppressed" for result in results))
        self.assertEqual(results[9].signals[0].type, "ALTITUDE_REVERSAL")

    def test_gap_ground_and_flight_change_reset_history(self) -> None:
        for reset_kind in ("gap", "ground", "flight"):
            engine = AircraftWarningEngine("speed_loss")
            for index in range(10):
                engine.update(state("aaaaaa", index * 30.0, 200 - 5 * index))
            if reset_kind == "gap":
                result = engine.update(state("aaaaaa", 500, 100))
            elif reset_kind == "ground":
                result = engine.update(state("aaaaaa", 300, 100, on_ground=True))
            else:
                result = engine.update(state("aaaaaa", 300, 100, flight_id="FLT2"))
            self.assertEqual(result.status, "collecting_history")
            self.assertFalse(result.alert)

    def test_out_of_order_rejected(self) -> None:
        engine = AircraftWarningEngine()
        engine.update(state("aaaaaa", 100, 150))
        with self.assertRaises(ValueError):
            engine.update(state("aaaaaa", 99, 145))

    def test_mapping_io_and_stale_history_reset(self) -> None:
        engine = AircraftWarningEngine()
        for index in range(10):
            engine.update(state("aaaaaa", index * 30.0, 150))
        result = engine.update({
            "timestamp": 300.0, "icao24": "AAAAAA", "flight_id": "FLT1",
            "position": {"altitude_baro_ft": 10000.0, "on_ground": False, "stale": True},
            "kinematics": {"ground_speed_kts": 100.0},
            "status": {"squawk": "7700", "emergency": "general"},
        })
        self.assertEqual(result.to_mapping()["status"], "collecting_history")
        self.assertFalse(result.alert)
        self.assertEqual(engine.update(state("aaaaaa", 330, 95)).status, "collecting_history")

    def test_replay_counts_early_target_and_late_predeclaration_warnings(self) -> None:
        events = {name: {"event_id": name, "icao24": name, "event_unix_s": 2000.0,
                         "signal_subtypes": ["emergency_squawk:7700"]}
                  for name in ("early", "target", "late")}
        alerts = [
            {"icao24": "early", "timestamp": 1100.0, "source_cohort": "candidate", "signals": []},
            {"icao24": "target", "timestamp": 1700.0, "source_cohort": "candidate", "signals": []},
            {"icao24": "late", "timestamp": 1970.0, "source_cohort": "candidate", "signals": []},
            {"icao24": "late", "timestamp": 2001.0, "source_cohort": "candidate", "signals": []},
        ]
        report = summarize_mode(alerts, events, 1.0)
        self.assertEqual(report["detected_events"], 3)
        self.assertEqual(report["strict_2_to_10_minute_events"], 1)
        self.assertEqual(report["warning_band_counts"], {
            "early_10_to_20_minutes_only": 1,
            "target_2_to_10_minutes": 1,
            "late_under_2_minutes_only": 1,
        })
        self.assertEqual(report["candidate_alerts_outside_20_minute_predeclaration_window"], 1)


if __name__ == "__main__":
    unittest.main()
