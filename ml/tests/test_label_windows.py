import argparse
import unittest

from ml.features import FEATURE_NAMES, extract_features
from ml.label_windows import labeled_rows
from ml.predict import probability


def point(timestamp, flight_leg_index=0):
    return {
        "timestamp_unix_s": float(timestamp),
        "flight_leg_index": flight_leg_index,
        "altitude_baro_ft": 10000.0 - timestamp,
        "ground_speed_kt": 220.0,
        "vertical_rate_fpm": -600.0,
        "track_deg": 90.0,
        "on_ground": False,
        "extra": {"emergency": "general", "squawk": "7700"},
    }


CONFIG = argparse.Namespace(
    context_seconds=300.0,
    min_lead_seconds=120.0,
    horizon_seconds=600.0,
    stride_seconds=60.0,
    max_gap_seconds=120.0,
    min_points=5,
)
RECORD = {"date_utc": "2024-02-15", "icao24": "abcdef", "trace_file": "trace.json.gz"}


class LabelWindowTests(unittest.TestCase):
    def test_positive_windows_stop_before_min_lead_and_exclude_status_features(self):
        points = [point(t) for t in range(0, 1801, 30)]
        examples = list(labeled_rows(points, 1200.0, RECORD, CONFIG))
        positive_anchors = [row["anchor_unix_s"] for row in examples if row["label"] == 1]
        self.assertEqual(positive_anchors, list(range(600, 1080, 60)))
        self.assertTrue(all(row["anchor_unix_s"] < 1080 for row in examples))
        self.assertEqual(tuple(examples[0]["features"]), FEATURE_NAMES)
        self.assertFalse(any("emergency" in name or "squawk" in name for name in FEATURE_NAMES))

    def test_gap_censors_windows_and_control_requires_future_coverage(self):
        short_control = [point(t) for t in range(0, 901, 30)]
        examples = list(labeled_rows(short_control, None, RECORD, CONFIG))
        self.assertEqual([row["anchor_unix_s"] for row in examples], [300.0])
        interrupted = [point(t) for t in range(0, 601, 30)] + [point(t) for t in range(900, 1801, 30)]
        self.assertFalse(any(row["label"] == 1 for row in labeled_rows(interrupted, 1200.0, RECORD, CONFIG)))

    def test_exported_logistic_formula(self):
        model = {
            "feature_names": ["x"], "coefficients": [2.0],
            "mean": [1.0], "scale": [2.0], "intercept": 0.0,
        }
        self.assertAlmostEqual(probability(model, {"x": 1.0}), 0.5)
        self.assertGreater(probability(model, {"x": 3.0}), 0.8)
        with self.assertRaises(ValueError):
            probability(model, {"x": 1.0, "squawk": 7700})


if __name__ == "__main__":
    unittest.main()
