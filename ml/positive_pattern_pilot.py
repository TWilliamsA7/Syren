"""Positive-first, one-date pilot for recurring ADS-B behavior warnings.

Patterns and their cutoffs are discovered on nine development dates. The tenth
date is replayed once, without fitting or selecting on its event labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.event_supervised import load_compact_dataset, pilot_split, replay_validation
from ml.features import TEMPORAL_FEATURE_NAMES
from scripts.fetch_event_traces import load_raw_trace, trace_points

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/learning/ten_day_development/declaration_episode_windows.jsonl"
COLLECTION = ROOT / "data/collection"
OUTPUT = ROOT / "data/learning/ten_day_development/positive_pattern_pilot"
HOLDOUT_DATE = "2026-05-15"
FEATURE = {name: index for index, name in enumerate(TEMPORAL_FEATURE_NAMES)}
MIN_EVENT_SUPPORT = 5
MIN_DATE_SUPPORT = 3
MIN_AIRCRAFT_SUPPORT = 5
QUANTILES = (0.5, 0.75)


def pattern_values(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Signed strength; larger always means more of the named behavior."""
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite temporal input")
    f = lambda name: matrix[:, FEATURE[name]].astype(np.float64)
    airborne = f("airborne_at_anchor") >= 0.5
    rate = airborne & (f("vertical_rate_valid_fraction") >= 0.5)
    speed = airborne & (f("speed_valid_fraction") >= 0.5)
    track = airborne & (f("track_valid_fraction") >= 0.5)
    altitude = airborne & (f("altitude_valid_fraction") >= 0.5)
    def guarded(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        return np.where(eligible, values, -np.inf)
    return {
        "sustained_descent": guarded(-f("vertical_rate_mean_fpm_60s"), rate),
        "worsening_descent": guarded(f("vertical_rate_mean_fpm_180s") - f("vertical_rate_mean_fpm_30s"), rate),
        "sustained_climb": guarded(f("vertical_rate_mean_fpm_60s"), rate),
        "speed_loss": guarded(-f("speed_trend_kt_s_60s"), speed),
        "speed_gain": guarded(f("speed_trend_kt_s_60s"), speed),
        "sustained_turn": guarded(f("turn_rate_deg_s_60s"), track),
        "turn_intensification": guarded(f("turn_rate_deg_s_30s") - f("turn_rate_deg_s_180s"), track),
        "altitude_reversal": guarded(f("altitude_range_ft") - abs(f("altitude_delta_ft")), altitude),
        "telemetry_gap": guarded(f("max_observation_gap_s"), airborne),
    }


def event_catalog(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, dict[str, Any]]:
    events = {}
    for index in indices:
        row = rows[index]
        for event in row["future_events"]:
            events.setdefault(event["event_id"], {
                "event_id": event["event_id"], "date_utc": row["date_utc"],
                "icao24": row["icao24"], "event_unix_s": float(event["event_unix_s"]),
                "signal_subtypes": event["signal_subtypes"],
            })
    return events


def discover_candidates(
    rows: list[dict[str, Any]], training: list[int], values: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Derive thresholds from per-event positive peaks, never held-out labels."""
    events = event_catalog(rows, training)
    event_indices: dict[str, list[int]] = defaultdict(list)
    for index in training:
        for event in rows[index]["future_events"]:
            event_indices[event["event_id"]].append(index)
    candidates = []
    for name, strength in values.items():
        peaks = {event_id: float(np.max(strength[indices])) for event_id, indices in event_indices.items()}
        finite_peaks = [peak for peak in peaks.values() if math.isfinite(peak)]
        if len(finite_peaks) < MIN_EVENT_SUPPORT:
            continue
        thresholds = sorted(set(float(np.quantile(finite_peaks, q)) for q in QUANTILES))
        for threshold in thresholds:
            # Zero-strength cutoffs mean there is no directional change to explain.
            if threshold <= 0:
                continue
            supported = [events[event_id] for event_id, peak in peaks.items() if peak >= threshold]
            dates = {event["date_utc"] for event in supported}
            aircraft = {event["icao24"] for event in supported}
            if (len(supported) < MIN_EVENT_SUPPORT or len(dates) < MIN_DATE_SUPPORT
                    or len(aircraft) < MIN_AIRCRAFT_SUPPORT):
                continue
            candidates.append({
                "pattern": name, "threshold": threshold,
                "discovery_event_support": len(supported),
                "discovery_date_support": len(dates),
                "discovery_aircraft_support": len(aircraft),
                "discovery_event_ids": sorted(event["event_id"] for event in supported),
            })
    return candidates


def replay_pattern(
    rows: list[dict[str, Any]], indices: list[int], strength: np.ndarray, threshold: float,
) -> dict[str, Any]:
    # A thresholded 60–180 second feature is already a sustained behavior.
    active = np.where(strength[indices] >= threshold, 1.0, 0.0)
    return replay_validation(rows, indices, active, 1.0)


def compact_replay(replay: dict[str, Any]) -> dict[str, Any]:
    return {key: replay[key] for key in (
        "event_count", "detected_events", "detected_aircraft", "event_recall",
        "warning_time_median_seconds", "control_hours", "control_false_alerts",
        "control_false_alerts_per_1000_hours", "candidate_negative_alerts",
    )}


def subtype_recall(replay: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    catalog = replay["matched_events"] + replay["missed_events"]
    matched = {item["event_id"] for item in replay["matched_events"]}
    subtypes = sorted({name for item in catalog for name in item["signal_subtypes"]})
    return {name: {
        "events": sum(name in item["signal_subtypes"] for item in catalog),
        "detected": sum(name in item["signal_subtypes"] and item["event_id"] in matched for item in catalog),
    } for name in subtypes}


def make_casebook(
    rows: list[dict[str, Any]], indices: list[int], output: Path,
    collection: Path = COLLECTION,
) -> None:
    """Save raw pre-declaration flight timelines for review, without status leakage."""
    manifests = {}
    for path in sorted((collection / "emergency_candidates").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            manifests[(record["date_utc"], record["icao24"])] = record
    events = event_catalog(rows, indices)
    with output.open("w", encoding="utf-8") as stream:
        for event in sorted(events.values(), key=lambda item: (item["date_utc"], item["icao24"], item["event_unix_s"])):
            manifest = manifests.get((event["date_utc"], event["icao24"]))
            if manifest is None:
                raise ValueError(f"Missing trace manifest for {event['event_id']}")
            raw = load_raw_trace(collection / manifest["trace_file"])
            timeline = []
            for point in trace_points(raw, event["icao24"]):
                lead = event["event_unix_s"] - point["timestamp_unix_s"]
                if 0 < lead <= 600:
                    timeline.append({
                        "seconds_before_declaration": round(lead, 2),
                        "altitude_baro_ft": point["altitude_baro_ft"],
                        "ground_speed_kt": point["ground_speed_kt"],
                        "vertical_rate_fpm": point["vertical_rate_fpm"],
                        "track_deg": point["track_deg"],
                        "on_ground": point["on_ground"],
                    })
            stream.write(json.dumps({**event, "pre_declaration_trace": timeline}, allow_nan=False) + "\n")


def baseline_at_rate(date: str, rate: float, source: Path) -> dict[str, Any]:
    if not source.is_file():
        return {"available": False}
    report = json.loads(source.read_text(encoding="utf-8"))
    result = {}
    for name in ("isolation_forest", "temporal_boosted_tree_baseline"):
        day = next(row for row in report[name]["per_date_results"] if row["date_utc"] == date)
        eligible = [point for point in day["alert_curve"]
                    if point["control_false_alerts_per_1000_hours"] is not None
                    and point["control_false_alerts_per_1000_hours"] <= rate + 1e-9]
        best = max(eligible, key=lambda point: point["detected_events"]) if eligible else None
        result[name] = ({"detected_events": best["detected_events"],
                         "control_false_alerts_per_1000_hours": best["control_false_alerts_per_1000_hours"]}
                        if best else {"detected_events": 0, "control_false_alerts_per_1000_hours": 0.0})
    return result


def run_pilot(dataset: Path = DATASET, output_dir: Path = OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, rows = load_compact_dataset(dataset, dataset.parent / "event_supervised/compact_episode_cache.joblib")
    training, validation = pilot_split(rows, HOLDOUT_DATE)
    values = pattern_values(matrix)
    candidates = discover_candidates(rows, training, values)
    if not candidates:
        raise ValueError("No recurring positive pattern met support criteria")
    for candidate in candidates:
        replay = replay_pattern(rows, training, values[candidate["pattern"]], candidate["threshold"])
        candidate["training_replay"] = compact_replay(replay)
        rate = replay["control_false_alerts_per_1000_hours"]
        candidate["training_priority"] = replay["detected_events"] / math.sqrt(1.0 + rate)
    candidates.sort(key=lambda item: item["training_priority"], reverse=True)
    # Keep the best threshold in each named behavior family for the union curve.
    selected = []
    seen = set()
    for candidate in candidates:
        if candidate["pattern"] not in seen:
            selected.append(candidate)
            seen.add(candidate["pattern"])
    individual = []
    baseline_source = dataset.parent / "temporal_episode_comparison/temporal_model_comparison.json"
    for candidate in candidates:
        replay = replay_pattern(rows, validation, values[candidate["pattern"]], candidate["threshold"])
        individual.append({"pattern": candidate["pattern"], "threshold": candidate["threshold"],
                           "validation": compact_replay(replay), "subtype_recall": subtype_recall(replay),
                           "matched_baselines": baseline_at_rate(
                               HOLDOUT_DATE, replay["control_false_alerts_per_1000_hours"], baseline_source)})
    curve = [{"number_of_patterns": 0, "included_patterns": [],
              "validation": compact_replay(replay_validation(
                  rows, validation, np.zeros(len(validation)), 1.0)),
              "subtype_recall": {}, "matched_baselines": baseline_at_rate(HOLDOUT_DATE, 0.0, baseline_source),
              "matched_events": [], "missed_events": list(event_catalog(rows, validation).values()),
              "control_false_alert_examples": []}]
    active = np.zeros(len(validation), dtype=bool)
    validation_by_example = {rows[index]["example_id"]: index for index in validation}
    for rank, candidate in enumerate(selected, 1):
        active |= values[candidate["pattern"]][validation] >= candidate["threshold"]
        replay = replay_validation(rows, validation, active.astype(float), 1.0)
        rate = replay["control_false_alerts_per_1000_hours"]
        curve.append({"number_of_patterns": rank, "included_patterns": [item["pattern"] for item in selected[:rank]],
                      "validation": compact_replay(replay), "subtype_recall": subtype_recall(replay),
                      "matched_baselines": baseline_at_rate(HOLDOUT_DATE, rate, baseline_source),
                      "matched_events": replay["matched_events"],
                      "missed_events": replay["missed_events"],
                      "control_false_alert_examples": [
                          {**example, "triggered_patterns": [
                              item["pattern"] for item in selected[:rank]
                              if values[item["pattern"]][validation_by_example[example["example_id"]]] >= item["threshold"]]}
                          for example in replay["control_false_alert_examples"][:20]]})
    make_casebook(rows, list(range(len(rows))), output_dir / "positive_event_casebook.jsonl")
    report = {
        "pilot_only": True, "validation_date": HOLDOUT_DATE,
        "training_dates": sorted({rows[index]["date_utc"] for index in training}),
        "training_aircraft_purged_from_validation": True,
        "training_events": len(event_catalog(rows, training)),
        "validation_events": len(event_catalog(rows, validation)),
        "candidate_patterns": candidates, "individual_validation_results": individual,
        "union_tradeoff_curve": curve,
        "casebook": "positive_event_casebook.jsonl",
        "warning": "All ten dates are development data. This one-date pilot selects no operational threshold and makes no future-date claim.",
    }
    (output_dir / f"pilot_{HOLDOUT_DATE}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run_pilot(args.dataset, args.output_dir)
    print(json.dumps({"candidate_patterns": len(result["candidate_patterns"]),
                      "union_curve": [{"patterns": item["number_of_patterns"],
                                       "events": item["validation"]["detected_events"],
                                       "false_alerts_per_1000h": item["validation"]["control_false_alerts_per_1000_hours"]}
                                      for item in result["union_tradeoff_curve"]]}))
