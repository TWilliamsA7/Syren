import unittest
import tempfile
import json
from pathlib import Path

from ml.evaluate_alerts import conservative_threshold, replay
from ml.features import TEMPORAL_FEATURE_NAMES, extract_temporal_features
from ml.train_baseline import group_weights
from ml.discover_events import discover
from ml.label_incidents import robust_label
from ml.bootstrap_report import bootstrap
from ml.audit_candidates import usable_signal
from ml.compare_models import select_development_rows


def point(t, altitude=10000, speed=200, rate=-500):
    return {
        "timestamp_unix_s": float(t), "altitude_baro_ft": altitude,
        "ground_speed_kt": speed, "vertical_rate_fpm": rate,
        "track_deg": 90.0, "on_ground": False,
        "extra": {"emergency": "general", "squawk": "7700"},
    }


class PipelineTests(unittest.TestCase):
    def test_temporal_features_are_causal_and_exclude_status(self):
        history = [point(t, 10000 - t, 200 + t / 10) for t in range(0, 301, 30)]
        features = extract_temporal_features(history)
        self.assertEqual(tuple(features), TEMPORAL_FEATURE_NAMES)
        self.assertLess(features["altitude_trend_ft_s_300s"], 0)
        self.assertGreater(features["speed_trend_kt_s_300s"], 0)
        self.assertFalse(any("squawk" in name or "emergency" in name for name in features))
        later = point(330, 0, 0)
        self.assertEqual(features, extract_temporal_features(history))
        self.assertNotEqual(features, extract_temporal_features(history + [later]))

    def test_audit_distinguishes_medical_priority_from_aircraft_distress(self):
        self.assertFalse(usable_signal({"signal": "emergency_field", "value": "lifeguard"}))
        self.assertTrue(usable_signal({"signal": "emergency_field", "value": "general"}))
        self.assertTrue(usable_signal({"signal": "emergency_squawk", "value": "7700"}))

    def test_new_dates_are_excluded_from_development_by_default(self):
        rows = [{"date_utc": day} for day in
                ("2024-02-15", "2024-05-15", "2024-08-15", "2025-01-01")]
        selected = select_development_rows(rows, ("2024-02-15", "2024-05-15", "2024-08-15"))
        self.assertEqual(len(selected), 3)
        self.assertNotIn("2025-01-01", {row["date_utc"] for row in selected})

    def test_group_weights_are_normalized_and_class_balanced(self):
        rows = [
            {"date_utc": "2024-01-01", "icao24": "aaaaaa", "label": 1},
            {"date_utc": "2024-01-01", "icao24": "aaaaaa", "label": 1},
            {"date_utc": "2024-01-01", "icao24": "bbbbbb", "label": 0},
            {"date_utc": "2024-01-01", "icao24": "cccccc", "label": 0},
        ]
        weights = group_weights(rows)
        self.assertAlmostEqual(sum(weights), len(rows))
        self.assertAlmostEqual(sum(w for w, r in zip(weights, rows) if r["label"]), 2)
        self.assertAlmostEqual(sum(w for w, r in zip(weights, rows) if not r["label"]), 2)

    def test_alert_replay_deduplicates_and_uses_control_airborne_hours(self):
        rows = [
            {"date_utc": "2024-01-01", "icao24": "aaaaaa", "group_id": "a",
             "anchor_unix_s": 0, "event_unix_s": 500, "label": 1,
             "source_cohort": "candidate", "airborne_exposure_seconds": 60},
            {"date_utc": "2024-01-01", "icao24": "aaaaaa", "group_id": "a",
             "anchor_unix_s": 60, "event_unix_s": 500, "label": 1,
             "source_cohort": "candidate", "airborne_exposure_seconds": 60},
            {"date_utc": "2024-01-01", "icao24": "bbbbbb", "group_id": "b",
             "anchor_unix_s": 0, "event_unix_s": None, "label": 0,
             "source_cohort": "control_sample", "airborne_exposure_seconds": 3600},
        ]
        result = replay(rows, [0.8, 0.9, 0.7], 0.6)
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["control_false_alerts"], 1)
        self.assertEqual(result["warning_seconds"], [500])
        self.assertEqual(result["control_observed_airborne_hours"], 1)
        self.assertGreater(conservative_threshold(rows, [0.8, 0.9, 0.7]), 0.7)

    def test_ntsb_discovery_stages_candidates_without_inventing_utc_time(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ntsb.csv"
            output = Path(directory) / "candidates.jsonl"
            source.write_text(
                "Accident.Number,Event.Date,Registration.Number,Investigation.Type\n"
                "ERA24LA123,02/15/2024,N123AB,Incident\n", encoding="utf-8"
            )
            result = discover(source, output)
            self.assertEqual(result["candidate_count"], 1)
            self.assertIn("pending_utc_time", output.read_text(encoding="utf-8"))
            self.assertNotIn("event_time_utc", output.read_text(encoding="utf-8"))

    def test_incident_time_uncertainty_censors_boundary_windows(self):
        self.assertEqual(robust_label(0, 500, 30), 1)
        self.assertEqual(robust_label(0, 700, 30), 0)
        self.assertIsNone(robust_label(0, 600, 30))
        self.assertIsNone(robust_label(0, 120, 30))

    def test_cluster_bootstrap_reads_oof_alert_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison.json").write_text(json.dumps({
                "best_exploratory": {"alert_at_conservative_1_per_1000h_threshold": {"threshold": 0.5}}
            }), encoding="utf-8")
            examples = [
                {"date_utc": "2024-01-01", "icao24": "aaaaaa", "group_id": "a",
                 "anchor_unix_s": 0, "event_unix_s": 500, "label": 1,
                 "source_cohort": "candidate", "airborne_exposure_seconds": 60, "score": 0.8},
                {"date_utc": "2024-01-01", "icao24": "bbbbbb", "group_id": "b",
                 "anchor_unix_s": 0, "event_unix_s": None, "label": 0,
                 "source_cohort": "control_sample", "airborne_exposure_seconds": 3600, "score": 0.2},
            ]
            (root / "best_oof_predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8"
            )
            output = bootstrap(root, iterations=10)
            self.assertEqual(output["iterations"], 10)
            self.assertTrue((root / "cluster_uncertainty.json").exists())


if __name__ == "__main__":
    unittest.main()
