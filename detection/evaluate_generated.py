"""Score generated telemetry against detector types the protocol can observe.

Run ``python -m detection.evaluate_generated data/generated/train_001`` after
``simulation.generate``. Fleet conflict screening is not part of this replay.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from detection.engine import build_default_engine


# These injections have a directly observable counterpart in FlightState.
SCORED_LABELS = {
    "rapid_descent": "RAPID_DESCENT",
    "erratic_altitude": "ALTITUDE_ANOMALY",
    "speed_loss": "SPEED_ANOMALY",
    "low_altitude_overspeed": "AGGRESSIVE_NEAR_GROUND_SPEED",
    "squawk": "EMERGENCY_SQUAWK",
    "signal_loss": "TELEMETRY_GAP",
}

# These are causes inferred from symptoms, rather than directly measured types.
PROXY_LABELS = {"engine_failure", "hijack"}

# Position and track show the flown path, but FlightState has no intended route.
UNSUPPORTED_LABELS = {"route_deviation"}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Replay one generated dataset and return class-aware, JSON-ready metrics."""
    scenario = json.loads((dataset_dir / "scenario.json").read_text(encoding="utf-8"))
    truth = json.loads((dataset_dir / "truth.json").read_text(encoding="utf-8"))
    events = {entry["icao24"]: entry for entry in truth}
    if len(events) != len(truth):
        raise ValueError("generated truth must contain at most one event per aircraft")
    known_labels = set(SCORED_LABELS) | PROXY_LABELS | UNSUPPORTED_LABELS
    if any(event["type"] not in known_labels for event in truth):
        raise ValueError("generated truth contains an unmapped event type")

    controls = {flight["icao24"] for flight in scenario["flights"]} - set(events)
    event_totals = Counter(event["type"] for event in truth)
    label_states: Counter[str] = Counter()
    true_states: Counter[str] = Counter()
    false_states: Counter[str] = Counter()
    event_hits: set[str] = set()
    proxy_hits: set[str] = set()
    first_latency: dict[str, list[float]] = defaultdict(list)
    proxy_output_types: dict[str, Counter[str]] = defaultdict(Counter)
    control_flights_alerted: set[str] = set()
    control_alert_flights_by_type: dict[str, set[str]] = defaultdict(set)
    normal_alert_states_by_type: Counter[str] = Counter()
    normal_state_alerts = 0
    states = 0
    engine = build_default_engine()
    start = time.perf_counter()

    with gzip.open(dataset_dir / "states.jsonl.gz", "rt", encoding="utf-8") as states_file, \
            gzip.open(dataset_dir / "labels.jsonl.gz", "rt", encoding="utf-8") as labels_file:
        for number, (state_line, label_line) in enumerate(zip_longest(states_file, labels_file), 1):
            if state_line is None or label_line is None:
                raise ValueError("generated state and label streams have different lengths")
            state = json.loads(state_line)
            label_record = json.loads(label_line)
            label = label_record["label"]
            if label != "normal" and label not in known_labels:
                raise ValueError(f"unmapped label {label!r} at state {number}")
            if (state["icao24"], state["timestamp"]) != (label_record["icao24"], label_record["timestamp"]):
                raise ValueError(f"state/label mismatch at row {number}")
            result = engine.update(state)
            output_types = {anomaly.type.value for anomaly in result.anomalies}
            states += 1
            label_states[label] += 1
            if states % 250_000 == 0:
                print(f"Processed {states:,} generated states", file=sys.stderr, flush=True)

            if label == "normal":
                if output_types:
                    normal_state_alerts += 1
                    normal_alert_states_by_type.update(output_types)
                    if state["icao24"] in controls:
                        control_flights_alerted.add(state["icao24"])
                        for output_type in output_types:
                            control_alert_flights_by_type[output_type].add(state["icao24"])

            if label == "normal" or label in SCORED_LABELS:
                for predicted_label, expected_type in SCORED_LABELS.items():
                    if expected_type in output_types:
                        if label == predicted_label:
                            true_states[predicted_label] += 1
                        else:
                            false_states[predicted_label] += 1

            if label in SCORED_LABELS and SCORED_LABELS[label] in output_types:
                if state["icao24"] not in event_hits:
                    event_hits.add(state["icao24"])
                    observed_at = state["timestamp"] + (state["status"].get("seen_age_s") or 0)
                    first_latency[label].append(observed_at - events[state["icao24"]]["start_time"])
            elif label in PROXY_LABELS and output_types:
                proxy_output_types[label].update(output_types)
                proxy_hits.add(state["icao24"])

    elapsed = time.perf_counter() - start
    scored: dict[str, dict[str, Any]] = {}
    for label in SCORED_LABELS:
        total = event_totals[label]
        hit_count = sum(1 for icao24 in event_hits if events[icao24]["type"] == label)
        tp = true_states[label]
        fp = false_states[label]
        fn = label_states[label] - tp
        scored[label] = {
            "expected_detector_type": SCORED_LABELS[label],
            "events_detected": hit_count,
            "events_total": total,
            "event_recall": _rate(hit_count, total),
            "median_first_alert_s": statistics.median(first_latency[label]) if first_latency[label] else None,
            "true_positive_states": tp,
            "false_positive_states": fp,
            "false_negative_states": fn,
            "state_precision": _rate(tp, tp + fp),
            "state_recall": _rate(tp, tp + fn),
        }

    return {
        "dataset": str(dataset_dir),
        "flights": len(scenario["flights"]),
        "states": states,
        "normal_states": label_states["normal"],
        "normal_state_alerts": normal_state_alerts,
        "normal_state_alert_rate": _rate(normal_state_alerts, label_states["normal"]),
        "normal_alert_states_by_type": dict(normal_alert_states_by_type),
        "control_flights": len(controls),
        "control_flights_alerted": len(control_flights_alerted),
        "control_flight_alert_rate": _rate(len(control_flights_alerted), len(controls)),
        "control_alert_flights_by_type": {
            output_type: len(identifiers)
            for output_type, identifiers in control_alert_flights_by_type.items()
        },
        "scored_types": scored,
        "proxy_only": {
            label: {
                "events_with_any_symptom_alert": sum(1 for icao24 in proxy_hits if events[icao24]["type"] == label),
                "events_total": event_totals[label],
                "output_type_counts": dict(proxy_output_types[label]),
            }
            for label in sorted(PROXY_LABELS)
        },
        "unsupported": {
            label: {"events": event_totals[label], "states_excluded": label_states[label]}
            for label in sorted(UNSUPPORTED_LABELS)
        },
        "performance": {"replay_seconds": elapsed, "states_per_second": states / elapsed if elapsed else None},
        "fleet_conflict_scan": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=Path("data/generated/train_001"))
    args = parser.parse_args()
    print(json.dumps(evaluate_dataset(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
