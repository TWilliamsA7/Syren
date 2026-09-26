"""Single-date pilot of positive patterns compared with similar control flights.

The previous 2026-05-15 pilot is diagnostic input. All pattern selection uses
other dates; 2024-11-15 is replayed once after selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.event_supervised import load_compact_dataset, pilot_split, replay_validation, sample_training_windows
from ml.features import TEMPORAL_FEATURE_NAMES
from ml.positive_pattern_pilot import (
    COLLECTION, DATASET, ROOT, baseline_at_rate, compact_replay, event_catalog,
    pattern_values, replay_pattern, subtype_recall,
)

HOLDOUT_DATE = "2024-11-15"
OUTPUT = ROOT / "data/learning/ten_day_development/contextual_pattern_pilot"
FEATURE = {name: index for index, name in enumerate(TEMPORAL_FEATURE_NAMES)}
MIN_CONTROL_WINDOWS = 50
MIN_CONTROL_AIRCRAFT = 8
MIN_EVENT_SUPPORT = 5
MIN_EVENT_DATES = 3
MIN_EVENT_AIRCRAFT = 5
MIN_TRAINING_ENRICHMENT = 2.0
POSITIVE_QUANTILES = (0.5, 0.75, 0.9)


def aircraft_types(collection: Path = COLLECTION) -> dict[tuple[str, str], str]:
    result = {}
    for directory in ("emergency_candidates", "control_manifests"):
        for path in sorted((collection / directory).glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                result[(row["date_utc"], row["icao24"])] = str(row.get("aircraft_type") or "unknown")
    return result


def contexts(matrix: np.ndarray, rows: list[dict[str, Any]], types: dict[tuple[str, str], str]) -> list[tuple[str, str, str, str]]:
    rate = matrix[:, FEATURE["vertical_rate_last_fpm"]]
    altitude = matrix[:, FEATURE["altitude_last_ft"]]
    speed = matrix[:, FEATURE["speed_last_kt"]]
    valid_rate = matrix[:, FEATURE["vertical_rate_available"]] >= 0.5
    phase = np.where(~valid_rate, "unknown", np.where(rate > 300, "climb", np.where(rate < -300, "descent", "level")))
    alt_band = np.where(altitude < 3000, "low", np.where(altitude < 10000, "mid", "high"))
    speed_band = np.where(speed < 150, "slow", np.where(speed < 300, "medium", "fast"))
    return [(str(phase[i]), str(alt_band[i]), str(speed_band[i]),
             types.get((row["date_utc"], row["icao24"]), "unknown"))
            for i, row in enumerate(rows)]


def context_groups(
    rows: list[dict[str, Any]], training_controls: list[int], context: list[tuple[str, str, str, str]],
) -> tuple[dict[tuple[str, ...], list[int]], list[tuple[str, ...]]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in training_controls:
        key = context[index]
        for length in (4, 3, 2, 1, 0):
            groups[key[:length]].append(index)
    eligible = {key for key, indices in groups.items()
                if len(indices) >= MIN_CONTROL_WINDOWS
                and len({rows[index]["icao24"] for index in indices}) >= MIN_CONTROL_AIRCRAFT}
    if () not in eligible:
        raise ValueError("Insufficient purged training controls for context matching")
    chosen = []
    for key in context:
        chosen.append(next(key[:length] for length in (4, 3, 2, 1, 0) if key[:length] in eligible))
    return groups, chosen


def contextual_rarity(
    strengths: dict[str, np.ndarray], groups: dict[tuple[str, ...], list[int]],
    chosen: list[tuple[str, ...]],
) -> dict[str, np.ndarray]:
    """Empirical upper-tail rarity against purged, sampled training controls."""
    selected: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(chosen):
        selected[key].append(index)
    result = {}
    for name, strength in strengths.items():
        rarity = np.zeros(len(strength), dtype=np.float32)
        for key, indices in selected.items():
            reference = np.sort(strength[groups[key]])
            reference = reference[np.isfinite(reference)]
            if len(reference) < MIN_CONTROL_WINDOWS:
                continue
            values = strength[indices]
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            target = np.asarray(indices)[finite]
            exceed = len(reference) - np.searchsorted(reference, values[finite], side="left")
            rarity[target] = -np.log10((exceed + 1) / (len(reference) + 1))
        result[name] = rarity
    return result


def review_prior_false_alerts(
    rows: list[dict[str, Any]], matrix: np.ndarray, strengths: dict[str, np.ndarray],
    context: list[tuple[str, str, str, str]],
) -> dict[str, Any]:
    indices = [index for index, row in enumerate(rows) if row["date_utc"] == "2026-05-15"]
    active = (strengths["altitude_reversal"][indices] >= 100).astype(float)
    replay = replay_validation(rows, indices, active, 1.0)
    by_example = {rows[index]["example_id"]: index for index in indices}
    false_indices = [by_example[item["example_id"]] for item in replay["control_false_alert_examples"]]
    counts = Counter(context[index][:2] for index in false_indices)
    values = strengths["altitude_reversal"][false_indices]
    return {
        "source": "2026-05-15 altitude_reversal >= 100 ft diagnostic replay",
        "control_false_alerts": len(false_indices),
        "phase_altitude_counts": {"/".join(key): count for key, count in sorted(counts.items())},
        "median_reversal_ft": float(np.median(values)),
        "median_speed_kt": float(np.median(matrix[false_indices, FEATURE["speed_last_kt"]])),
        "note": "This previously inspected date is not the new holdout.",
    }


def discover(
    rows: list[dict[str, Any]], training: list[int], rarity: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    catalog = event_catalog(rows, training)
    windows: dict[str, list[int]] = defaultdict(list)
    for index in training:
        for event in rows[index]["future_events"]:
            windows[event["event_id"]].append(index)
    candidates = []
    for name, values in rarity.items():
        peaks = {event_id: float(np.max(values[indices])) for event_id, indices in windows.items()}
        thresholds = sorted(set(float(np.quantile(list(peaks.values()), q)) for q in POSITIVE_QUANTILES))
        for threshold in thresholds:
            if threshold <= 0:
                continue
            supported = [catalog[event_id] for event_id, peak in peaks.items() if peak >= threshold]
            if (len(supported) < MIN_EVENT_SUPPORT
                    or len({event["date_utc"] for event in supported}) < MIN_EVENT_DATES
                    or len({event["icao24"] for event in supported}) < MIN_EVENT_AIRCRAFT):
                continue
            replay = replay_pattern(rows, training, values, threshold)
            rate = replay["control_false_alerts_per_1000_hours"]
            # An event has about eight minutes of eligible warning opportunity.
            control_chance = min(1.0, rate / 7500.0)
            enrichment = replay["event_recall"] / max(control_chance, 1e-9)
            candidates.append({
                "pattern": name, "context_rarity_cutoff": threshold,
                "positive_support_events": len(supported),
                "positive_support_dates": len({event["date_utc"] for event in supported}),
                "positive_support_aircraft": len({event["icao24"] for event in supported}),
                "training_enrichment_vs_control_8min": enrichment,
                "passes_training_enrichment": enrichment >= MIN_TRAINING_ENRICHMENT,
                "training_replay": compact_replay(replay),
                "training_priority": replay["detected_events"] / (1.0 + rate),
            })
    return sorted(candidates, key=lambda item: item["training_priority"], reverse=True)


def run_pilot(dataset: Path = DATASET, output_dir: Path = OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, rows = load_compact_dataset(dataset, dataset.parent / "event_supervised/compact_episode_cache.joblib")
    training, validation = pilot_split(rows, HOLDOUT_DATE)
    sampled = sample_training_windows(rows, training)
    train_controls = [index for index in sampled if rows[index]["source_cohort"] == "control_sample"]
    strength = pattern_values(matrix)
    context = contexts(matrix, rows, aircraft_types())
    groups, chosen = context_groups(rows, train_controls, context)
    rarity = contextual_rarity(strength, groups, chosen)
    review = review_prior_false_alerts(rows, matrix, strength, context)
    candidates = discover(rows, training, rarity)
    baseline_source = dataset.parent / "temporal_episode_comparison/temporal_model_comparison.json"
    individual = []
    for candidate in candidates:
        replay = replay_pattern(rows, validation, rarity[candidate["pattern"]], candidate["context_rarity_cutoff"])
        individual.append({"pattern": candidate["pattern"], "context_rarity_cutoff": candidate["context_rarity_cutoff"],
                           "validation": compact_replay(replay), "subtype_recall": subtype_recall(replay),
                           "matched_baselines": baseline_at_rate(
                               HOLDOUT_DATE, replay["control_false_alerts_per_1000_hours"], baseline_source)})
    selected = []
    seen = set()
    for candidate in candidates:
        if candidate["pattern"] not in seen:
            selected.append(candidate)
            seen.add(candidate["pattern"])
    curve = []
    active = np.zeros(len(validation), dtype=bool)
    for candidate in selected:
        active |= rarity[candidate["pattern"]][validation] >= candidate["context_rarity_cutoff"]
        replay = replay_validation(rows, validation, active.astype(float), 1.0)
        curve.append({
            "patterns": [item["pattern"] for item in selected[:len(curve) + 1]],
            "validation": compact_replay(replay), "subtype_recall": subtype_recall(replay),
            "matched_baselines": baseline_at_rate(
                HOLDOUT_DATE, replay["control_false_alerts_per_1000_hours"], baseline_source),
            "matched_events": replay["matched_events"], "missed_events": replay["missed_events"],
            "control_false_alert_examples": replay["control_false_alert_examples"][:20],
        })
    old = replay_pattern(rows, validation, strength["altitude_reversal"], 100.0)
    match_levels = Counter(len(chosen[index]) for index in validation)
    report = {
        "pilot_only": True, "holdout_date": HOLDOUT_DATE,
        "training_dates": sorted({rows[index]["date_utc"] for index in training}),
        "training_aircraft_purged_from_holdout": True,
        "training_events": len(event_catalog(rows, training)),
        "holdout_events": len(event_catalog(rows, validation)),
        "matched_control_reference_windows": len(train_controls),
        "holdout_context_match_levels": {
            "phase_altitude_speed_type": match_levels[4],
            "phase_altitude_speed": match_levels[3],
            "phase_altitude": match_levels[2],
            "phase": match_levels[1],
            "all_controls": match_levels[0],
        },
        "prior_false_alert_review": review,
        "context": "phase, altitude band, speed band, and aircraft type when >=50 control windows from >=8 aircraft; otherwise progressively broader context",
        "score": "upper-tail rarity of a named temporal behavior within matched training controls; not an emergency probability",
        "candidate_patterns": candidates, "individual_validation_results": individual,
        "patterns_passing_training_enrichment": sum(item["passes_training_enrichment"] for item in candidates),
        "union_tradeoff_curve": curve,
        "old_altitude_reversal_100ft_on_holdout": compact_replay(old),
        "old_altitude_reversal_matched_baselines": baseline_at_rate(
            HOLDOUT_DATE, old["control_false_alerts_per_1000_hours"], baseline_source),
        "warning": "Development-only one-date pilot. The prior 2026 date informed the method. No full training or operational threshold.",
    }
    (output_dir / f"pilot_{HOLDOUT_DATE}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run_pilot(args.dataset, args.output_dir)
    print(json.dumps({"candidates": len(result["candidate_patterns"]),
                      "holdout_events": result["holdout_events"],
                      "individual": [{"pattern": item["pattern"],
                                      "events": item["validation"]["detected_events"],
                                      "false_per_1000h": item["validation"]["control_false_alerts_per_1000_hours"]}
                                     for item in result["individual_validation_results"]],
                      "union": [{"patterns": len(item["patterns"]),
                                 "events": item["validation"]["detected_events"],
                                 "false_per_1000h": item["validation"]["control_false_alerts_per_1000_hours"]}
                                for item in result["union_tradeoff_curve"]]}))
