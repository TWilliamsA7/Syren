from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from detection.evaluate_generated import evaluate_dataset
from detection.harness import make_state


class GeneratedEvaluatorTests(unittest.TestCase):
    def test_scores_matching_types_and_excludes_unsupported_events(self) -> None:
        rows = [
            (make_state(timestamp=0, icao24="clean01"), "normal"),
            (make_state(timestamp=10, icao24="squawk01", squawk="7700"), "squawk"),
            (make_state(timestamp=20, icao24="speed01", squawk="7700"), "speed_loss"),
            (make_state(timestamp=30, icao24="route01", squawk="7700"), "route_deviation"),
            (make_state(timestamp=40, icao24="engine01", squawk="7700"), "engine_failure"),
        ]
        truth = [
            {"icao24": state["icao24"], "type": label, "start_time": state["timestamp"],
             "end_time": state["timestamp"] + 10}
            for state, label in rows if label != "normal"
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenario.json").write_text(json.dumps({"flights": [{"icao24": state["icao24"]} for state, _ in rows]}))
            (root / "truth.json").write_text(json.dumps(truth))
            with gzip.open(root / "states.jsonl.gz", "wt") as states_file, \
                    gzip.open(root / "labels.jsonl.gz", "wt") as labels_file:
                for state, label in rows:
                    states_file.write(json.dumps(state) + "\n")
                    labels_file.write(json.dumps({"icao24": state["icao24"], "timestamp": state["timestamp"], "label": label}) + "\n")
            report = evaluate_dataset(root)

        self.assertEqual(report["normal_state_alerts"], 0)
        self.assertEqual(report["control_flights"], 1)
        self.assertEqual(report["scored_types"]["squawk"]["events_detected"], 1)
        self.assertEqual(report["scored_types"]["squawk"]["false_positive_states"], 1)
        self.assertEqual(report["scored_types"]["speed_loss"]["events_detected"], 0)
        self.assertEqual(report["unsupported"]["route_deviation"], {"events": 1, "states_excluded": 1})
        self.assertEqual(report["proxy_only"]["engine_failure"]["events_with_any_symptom_alert"], 1)


if __name__ == "__main__":
    unittest.main()
