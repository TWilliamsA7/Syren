import tempfile
import unittest
from pathlib import Path

import numpy as np
import joblib

from ml.compare_temporal_models import (
    balanced_control_sample,
    curve_thresholds,
    fit_isolation_forest,
    isolation_anomaly_scores,
    purged_date_split,
    temporal_feature_matrix,
    alert_metrics,
)
from ml.features import TEMPORAL_FEATURE_NAMES
from ml.predict import isolation_anomaly_score


def row(day, aircraft, anchor, *, cohort="control_sample", label=0, altitude=10000.0):
    features = {name: 0.0 for name in TEMPORAL_FEATURE_NAMES}
    features["altitude_last_ft"] = altitude
    return {
        "date_utc": day,
        "icao24": aircraft,
        "group_id": f"{day}:{aircraft}:0",
        "anchor_unix_s": float(anchor),
        "source_cohort": cohort,
        "airborne_exposure_seconds": 60.0,
        "targets": {"any_declaration": label},
        "label": label,
        "temporal_features": features,
        "future_events": [],
    }


class IsolationForestTests(unittest.TestCase):
    def test_control_sample_caps_windows_per_aircraft_day_evenly(self):
        controls = [row("2024-01-01", "aaaaaa", t) for t in range(0, 700, 100)]
        selected = balanced_control_sample(controls, max_windows_per_aircraft_day=4)
        self.assertEqual([item["anchor_unix_s"] for item in selected], [0.0, 200.0, 400.0, 600.0])

    def test_fold_training_excludes_held_out_and_purged_aircraft(self):
        rows = [
            row("2024-01-01", "shared", 0),
            row("2024-01-02", "shared", 0, cohort="candidate", label=1),
            row("2024-01-01", "trainonly", 0),
            row("2024-01-01", "trainonly", 60),
        ]
        train, validation, purged = purged_date_split(rows, "2024-01-02")
        self.assertEqual(purged, {"shared"})
        self.assertEqual({item["icao24"] for item in validation}, {"shared"})
        model, sampled = fit_isolation_forest(train)
        self.assertTrue(sampled)
        self.assertTrue(all(item["date_utc"] != "2024-01-02" for item in sampled))
        self.assertTrue(all(item["icao24"] not in purged for item in sampled))
        self.assertTrue(all(item["source_cohort"] == "control_sample" for item in sampled))
        self.assertTrue(all(item["targets"]["any_declaration"] == 0 for item in sampled))
        self.assertEqual(model.n_estimators, 200)

    def test_higher_score_means_more_anomalous(self):
        normal = [
            row("2024-01-01", f"{index:06x}", 0, altitude=10000 + index)
            for index in range(40)
        ]
        model, _ = fit_isolation_forest(normal)
        expected = row("2024-02-01", "normal", 0, altitude=10010)
        outlier = row("2024-02-01", "outlier", 0, altitude=1_000_000)
        normal_score, outlier_score = isolation_anomaly_scores(model, [expected, outlier])
        self.assertGreater(float(outlier_score), float(normal_score))

    def test_feature_validation_rejects_nonfinite_and_boolean_values(self):
        item = row("2024-01-01", "aaaaaa", 0)
        item["temporal_features"]["altitude_last_ft"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            temporal_feature_matrix([item])
        item["temporal_features"]["altitude_last_ft"] = True
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            temporal_feature_matrix([item])

    def test_saved_bundle_inference_matches_in_memory_score(self):
        controls = [
            row("2024-01-01", f"{index:06x}", 0, altitude=10000 + index)
            for index in range(40)
        ]
        model, _ = fit_isolation_forest(controls)
        example = controls[0]
        bundle = {
            "model_type": "isolation_forest",
            "target": "future_adsb_declaration_episode_multilabel_2_to_10_minutes",
            "score_type": "isolation_anomaly_score",
            "feature_names": list(TEMPORAL_FEATURE_NAMES),
            "model": model,
        }
        before = isolation_anomaly_scores(model, [example])[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            after = isolation_anomaly_score(loaded, example["temporal_features"])
        self.assertAlmostEqual(float(before), after, places=12)

    def test_alert_curve_covers_raw_scores_outside_probability_range(self):
        thresholds = curve_thresholds([-0.25, 0.1, 0.35])
        self.assertEqual(thresholds[0], -0.25)
        self.assertGreater(thresholds[-1], 0.35)

    def test_episode_replay_reports_lead_and_suppresses_repeat_alert(self):
        event = {"event_id": "evt", "event_unix_s": 300.0, "signal_subtypes": ["emergency_field:general"]}
        candidate_a = row("2024-01-01", "aaaaaa", 0, cohort="candidate", label=1)
        candidate_a["group_id"] = "candidate-flight"
        candidate_a["future_events"] = [event]
        candidate_b = dict(candidate_a, anchor_unix_s=60.0)
        control = row("2024-01-01", "bbbbbb", 0)
        control["group_id"] = "control-flight"
        control["airborne_exposure_seconds"] = 3600.0

        result = alert_metrics([candidate_a, candidate_b, control], [0.2, 0.4, 0.3], 0.1)
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["warning_time_median_seconds"], 300.0)
        self.assertEqual(result["control_false_alerts"], 1)
        self.assertEqual(result["control_observed_airborne_hours"], 1.0)


if __name__ == "__main__":
    unittest.main()
