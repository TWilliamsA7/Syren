import unittest
from types import SimpleNamespace

from ml.compare_temporal_models import purged_date_split
from ml.label_windows import build_signal_episodes, labeled_rows
from ml.sequence_features import resample_sequence


def point(timestamp, altitude=1000.0, track=90.0, ground=False, latitude=40.0, longitude=-70.0):
    return {
        "timestamp_unix_s": float(timestamp),
        "latitude_deg": float(latitude),
        "longitude_deg": float(longitude),
        "altitude_baro_ft": float(altitude),
        "ground_speed_kt": 200.0,
        "vertical_rate_fpm": -300.0,
        "track_deg": float(track),
        "on_ground": ground,
        "flight_leg_index": 0,
    }


class SequenceFeatureTests(unittest.TestCase):
    def test_resampling_includes_history_start_and_anchor(self):
        rows = [point(t, altitude=t) for t in range(0, 301, 10)]
        sequence = resample_sequence(rows, 300)
        self.assertEqual((len(sequence), len(sequence[0])), (31, 14))
        self.assertEqual(sequence[0][0], 0.0)
        self.assertEqual(sequence[-1][0], 300.0)
        self.assertTrue(all(row[6] == 1.0 for row in sequence))

    def test_track_interpolation_wraps_across_zero_degrees(self):
        sequence = resample_sequence([point(0, track=359), point(20, track=1)], 20)
        halfway = sequence[29]
        self.assertEqual(halfway[9], 1.0)
        self.assertAlmostEqual(halfway[3], 0.0, places=7)
        self.assertAlmostEqual(halfway[4], 1.0, places=7)

    def test_large_gaps_preserve_missingness_masks(self):
        sequence = resample_sequence([point(0), point(40)], 40)
        self.assertEqual(sequence[26][6], 1.0)
        self.assertEqual(sequence[27][6], 0.0)
        self.assertEqual(sequence[28][6], 0.0)
        self.assertEqual(sequence[30][6], 1.0)
        self.assertEqual(sequence[27][10], 1.0)  # recent categorical state is carried causally

    def test_future_observation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Future observations"):
            resample_sequence([point(0), point(301)], 300)

    def test_anchor_relative_path_is_causal_and_centered(self):
        rows = [point(0, longitude=-70.02), point(20, longitude=-70.0)]
        sequence = resample_sequence(rows, 20)
        self.assertEqual(sequence[29][13], 1.0)
        self.assertLess(sequence[29][11], 0.0)
        self.assertAlmostEqual(sequence[30][11], 0.0)
        self.assertAlmostEqual(sequence[30][12], 0.0)
        self.assertEqual(sequence[30][13], 1.0)

    def test_position_interpolation_wraps_at_antimeridian(self):
        sequence = resample_sequence([
            point(0, longitude=179.9), point(20, longitude=-179.9),
        ], 20)
        self.assertEqual(sequence[29][13], 1.0)
        self.assertLess(abs(sequence[29][11]), 20.0)

    def test_signals_within_one_minute_form_multilabel_episode(self):
        episodes = build_signal_episodes(1000.0, [
            {"offset_s": 10.0, "signal": "emergency_field", "value": "general"},
            {"offset_s": 10.0, "signal": "emergency_squawk", "value": "7700"},
            {"offset_s": 65.0, "signal": "emergency_field", "value": "nordo"},
            {"offset_s": 131.0, "signal": "emergency_squawk", "value": "7600"},
        ], "2024-01-01", "abcdef")
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["signal_subtypes"], [
            "emergency_field:general", "emergency_field:nordo", "emergency_squawk:7700",
        ])
        self.assertEqual(episodes[1]["event_unix_s"], 1121.0)

    def test_episode_window_labels_keep_event_subtypes_at_each_onset(self):
        rows = [point(t, altitude=1000 + t, latitude=40 + t * 0.00001) for t in range(0, 1201, 10)]
        events = [
            {"event_id": "one", "event_unix_s": 600.0,
             "signal_subtypes": ["emergency_field:general"]},
            {"event_id": "two", "event_unix_s": 900.0,
             "signal_subtypes": ["emergency_squawk:7700"]},
        ]
        config = SimpleNamespace(
            max_gap_seconds=120.0, min_points=10, stride_seconds=60.0,
            context_seconds=300.0, min_lead_seconds=120.0, horizon_seconds=600.0,
        )
        record = {"date_utc": "2024-01-01", "icao24": "abcdef", "trace_file": "trace.json.gz"}
        output = list(labeled_rows(rows, None, record, config, event_episodes=events))
        by_anchor = {row["anchor_unix_s"]: row for row in output}
        self.assertEqual(by_anchor[300.0]["targets"]["any_declaration"], 1)
        self.assertEqual(by_anchor[300.0]["targets"]["emergency_field:general"], 1)
        self.assertEqual(by_anchor[300.0]["targets"]["emergency_squawk:7700"], 1)
        self.assertEqual([event["event_id"] for event in by_anchor[300.0]["future_events"]], ["one", "two"])
        self.assertNotIn(480.0, by_anchor)  # 120 seconds before event one is intentionally unlabeled.

    def test_fold_purges_aircraft_shared_with_validation_date(self):
        rows = [
            {"date_utc": "2024-01-01", "icao24": "shared", "label": 0},
            {"date_utc": "2024-01-02", "icao24": "shared", "label": 1},
            {"date_utc": "2024-01-01", "icao24": "train-only", "label": 1},
        ]
        train, validation, purged = purged_date_split(rows, "2024-01-02")
        self.assertEqual([row["icao24"] for row in train], ["train-only"])
        self.assertTrue(all(row["date_utc"] != "2024-01-02" for row in train))
        self.assertEqual({row["icao24"] for row in validation}, {"shared"})
        self.assertEqual(purged, {"shared"})


if __name__ == "__main__":
    unittest.main()
