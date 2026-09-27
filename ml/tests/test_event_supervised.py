import unittest

import numpy as np

from ml.event_supervised import (
    cutoff_from_training_controls,
    event_balanced_weights,
    pilot_split,
    replay_validation,
)


def row(day, aircraft, anchor, *, label=0, event_id=None, cohort="control_sample", group=None,
        exposure=60.0):
    events = ([{"event_id": event_id, "event_unix_s": 300.0,
                "signal_subtypes": ["emergency_field:general"]}] if event_id else [])
    return {
        "example_id": f"{day}:{aircraft}:{anchor}",
        "date_utc": day,
        "icao24": aircraft,
        "group_id": group or f"{day}:{aircraft}",
        "anchor_unix_s": float(anchor),
        "source_cohort": cohort,
        "airborne_exposure_seconds": exposure,
        "label": label,
        "future_events": events,
    }


class EventSupervisedTests(unittest.TestCase):
    def test_pilot_split_purges_validation_aircraft(self):
        rows = [
            row("2025-05-15", "shared", 0),
            row("2025-05-15", "train", 0),
            row("2025-08-15", "shared", 0, label=1, event_id="event", cohort="candidate"),
        ]
        training, validation = pilot_split(rows)
        self.assertEqual(training, [1])
        self.assertEqual(validation, [2])

    def test_each_real_event_and_negative_group_have_equal_mass(self):
        rows = [
            row("2025-05-15", "a", 0, label=1, event_id="event-a", cohort="candidate"),
            row("2025-05-15", "a", 60, label=1, event_id="event-a", cohort="candidate"),
            row("2025-05-15", "b", 0, label=1, event_id="event-b", cohort="candidate"),
            row("2025-05-15", "c", 0),
            row("2025-05-15", "c", 60),
            row("2025-05-15", "d", 0),
        ]
        weights = event_balanced_weights(rows, list(range(len(rows))))
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])
        self.assertAlmostEqual(weights[3] + weights[4], weights[5])
        self.assertAlmostEqual(sum(weights[:3]), sum(weights[3:]))

    def test_control_cutoff_applies_suppression_and_rejects_candidates(self):
        rows = [row("2025-05-15", "a", anchor, exposure=3600.0)
                for anchor in (0, 60, 700)]
        result = cutoff_from_training_controls(rows, [0, 1, 2], np.asarray([0.9, 0.8, 0.7]), 500.0)
        self.assertEqual(result["training_control_alert_limit"], 1)
        self.assertEqual(result["training_control_alerts"], 1)
        self.assertEqual(result["score_cutoff"], 0.8)
        rows[0]["source_cohort"] = "candidate"
        with self.assertRaisesRegex(ValueError, "only training controls"):
            cutoff_from_training_controls(rows, [0, 1, 2], np.asarray([0.9, 0.8, 0.7]), 500.0)

    def test_replay_matches_event_once_and_reports_warning(self):
        rows = [
            row("2025-08-15", "a", 0, label=1, event_id="event", cohort="candidate"),
            row("2025-08-15", "a", 60, label=1, event_id="event", cohort="candidate"),
            row("2025-08-15", "b", 0, exposure=3600.0),
        ]
        result = replay_validation(rows, [0, 1, 2], np.asarray([0.9, 0.8, 0.9]), 0.5)
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["matched_events"][0]["event_id"], "event")
        self.assertEqual(result["warning_time_median_seconds"], 300.0)
        self.assertEqual(result["control_false_alerts"], 1)
        self.assertEqual(result["control_hours"], 1.0)


if __name__ == "__main__":
    unittest.main()
