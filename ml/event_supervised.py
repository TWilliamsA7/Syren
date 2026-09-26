"""Fast, event-weighted pilot for the ADS-B declaration research target.

The pilot deliberately runs one held-out date before any ten-fold training.
Only real labeled events are positive examples. The cutoff is chosen from
training controls without inspecting validation labels.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.features import TEMPORAL_FEATURE_NAMES
from ml.label_windows import EPISODE_TARGET

PILOT_DATE = "2025-08-15"
SEED = 17
NEGATIVE_WINDOWS_PER_AIRCRAFT_DAY = 8
FALSE_ALERT_BUDGET_PER_1000H = 25.0
SUPPRESSION_SECONDS = 600.0
CACHE_VERSION = 1
MODEL_CONFIGS = {
    "event_weighted_hgb": {
        "max_iter": 60,
        "max_leaf_nodes": 7,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": SEED,
    },
    "event_weighted_extra_trees": {
        "n_estimators": 128,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": SEED,
    },
}


def load_compact_dataset(dataset: Path, cache: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Read the large JSONL once, keeping only causal features and replay fields."""
    source = dataset.stat()
    expected = {
        "version": CACHE_VERSION,
        "source_size": source.st_size,
        "source_mtime_ns": source.st_mtime_ns,
        "feature_names": list(TEMPORAL_FEATURE_NAMES),
    }
    if cache.is_file():
        saved = joblib.load(cache)
        if all(saved.get(key) == value for key, value in expected.items()):
            matrix, rows = saved["matrix"], saved["rows"]
            if matrix.shape == (len(rows), len(TEMPORAL_FEATURE_NAMES)):
                print(f"Loaded compact cache: {len(rows)} windows", flush=True)
                return matrix, rows

    values: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    with dataset.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            source_row = json.loads(line)
            if source_row.get("target") != EPISODE_TARGET:
                raise ValueError(f"Target mismatch at line {line_no}")
            features = source_row.get("temporal_features")
            if not isinstance(features, dict) or tuple(features) != TEMPORAL_FEATURE_NAMES:
                raise ValueError(f"Feature schema mismatch at line {line_no}")
            vector = []
            for name in TEMPORAL_FEATURE_NAMES:
                value = features[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Nonfinite feature {name} at line {line_no}")
                vector.append(float(value))
            label = source_row.get("label")
            events = source_row.get("future_events")
            if label not in (0, 1) or not isinstance(events, list) or bool(events) != bool(label):
                raise ValueError(f"Invalid event label at line {line_no}")
            rows.append({
                "example_id": source_row["example_id"],
                "date_utc": source_row["date_utc"],
                "icao24": source_row["icao24"],
                "group_id": source_row["group_id"],
                "anchor_unix_s": float(source_row["anchor_unix_s"]),
                "source_cohort": source_row["source_cohort"],
                "airborne_exposure_seconds": float(source_row["airborne_exposure_seconds"]),
                "label": int(label),
                "future_events": events,
            })
            values.append(vector)
    if not rows:
        raise ValueError("No labeled windows found")
    matrix = np.asarray(values, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**expected, "matrix": matrix, "rows": rows}, cache)
    print(f"Built compact cache: {len(rows)} windows", flush=True)
    return matrix, rows


def pilot_split(rows: list[dict[str, Any]], date: str = PILOT_DATE) -> tuple[list[int], list[int]]:
    validation = [index for index, row in enumerate(rows) if row["date_utc"] == date]
    if not validation:
        raise ValueError(f"No validation rows for {date}")
    aircraft = {rows[index]["icao24"] for index in validation}
    training = [index for index, row in enumerate(rows)
                if row["date_utc"] != date and row["icao24"] not in aircraft]
    if not training or {rows[index]["icao24"] for index in training} & aircraft:
        raise ValueError("Validation aircraft purge failed")
    return training, validation


def sample_training_windows(
    rows: list[dict[str, Any]], training: list[int],
    max_negative_per_aircraft_day: int = NEGATIVE_WINDOWS_PER_AIRCRAFT_DAY,
) -> list[int]:
    if max_negative_per_aircraft_day < 1:
        raise ValueError("Negative window cap must be positive")
    selected = [index for index in training if rows[index]["label"] == 1]
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index in training:
        row = rows[index]
        if row["label"] == 0:
            groups[(row["date_utc"], row["icao24"], row["source_cohort"])].append(index)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda index: rows[index]["anchor_unix_s"])
        if len(group) > max_negative_per_aircraft_day:
            positions = np.linspace(0, len(group) - 1, max_negative_per_aircraft_day, dtype=int)
            group = [group[position] for position in positions]
        selected.extend(group)
    return sorted(selected)


def event_balanced_weights(rows: list[dict[str, Any]], selected: list[int]) -> np.ndarray:
    """Give each real event equal positive mass and each negative group equal mass."""
    event_counts: Counter[str] = Counter()
    negative_counts: Counter[tuple[str, str, str]] = Counter()
    for index in selected:
        row = rows[index]
        if row["label"]:
            ids = {event["event_id"] for event in row["future_events"]}
            if not ids:
                raise ValueError("Positive window has no event ID")
            event_counts.update(ids)
        else:
            negative_counts[(row["date_utc"], row["icao24"], row["source_cohort"])] += 1
    if not event_counts or not negative_counts:
        raise ValueError("Pilot training needs positive events and negative groups")
    raw = np.zeros(len(selected), dtype=float)
    positive_mask = np.zeros(len(selected), dtype=bool)
    for position, index in enumerate(selected):
        row = rows[index]
        if row["label"]:
            positive_mask[position] = True
            raw[position] = sum(1.0 / event_counts[event_id]
                                for event_id in {event["event_id"] for event in row["future_events"]})
        else:
            key = (row["date_utc"], row["icao24"], row["source_cohort"])
            raw[position] = 1.0 / negative_counts[key]
    raw[positive_mask] *= 0.5 * len(selected) / raw[positive_mask].sum()
    raw[~positive_mask] *= 0.5 * len(selected) / raw[~positive_mask].sum()
    return raw


def build_model(name: str):
    if name == "event_weighted_hgb":
        return HistGradientBoostingClassifier(**MODEL_CONFIGS[name])
    if name == "event_weighted_extra_trees":
        return ExtraTreesClassifier(**MODEL_CONFIGS[name])
    raise ValueError(f"Unknown pilot model {name}")


def count_suppressed_alerts(anchors: list[float]) -> int:
    last = -math.inf
    count = 0
    for anchor in anchors:
        if anchor - last >= SUPPRESSION_SECONDS:
            count += 1
            last = anchor
    return count


def cutoff_from_training_controls(
    rows: list[dict[str, Any]], indices: list[int], scores: np.ndarray,
    budget_per_1000h: float = FALSE_ALERT_BUDGET_PER_1000H,
) -> dict[str, float | int]:
    """Find the most permissive control-only cutoff under a replayed alert budget."""
    if len(indices) != len(scores) or not len(indices) or not np.isfinite(scores).all():
        raise ValueError("Control scores must be finite and align with rows")
    if any(rows[index]["source_cohort"] != "control_sample" for index in indices):
        raise ValueError("Cutoff selection may use only training controls")
    hours = sum(rows[index]["airborne_exposure_seconds"] for index in indices) / 3600.0
    if hours <= 0:
        raise ValueError("Training controls have no airborne exposure")
    maximum_alerts = math.floor(budget_per_1000h * hours / 1000.0)
    ordered = sorted(zip((float(score) for score in scores), indices), reverse=True)
    active: dict[str, list[float]] = defaultdict(list)
    group_counts: dict[str, int] = {}
    alert_count = 0
    cutoff = math.nextafter(ordered[0][0], math.inf)
    selected_alerts = 0
    position = 0
    while position < len(ordered):
        score = ordered[position][0]
        while position < len(ordered) and ordered[position][0] == score:
            index = ordered[position][1]
            row = rows[index]
            group_id = row["group_id"]
            anchors = active[group_id]
            old_count = group_counts.get(group_id, 0)
            bisect.insort(anchors, row["anchor_unix_s"])
            new_count = count_suppressed_alerts(anchors)
            group_counts[group_id] = new_count
            alert_count += new_count - old_count
            position += 1
        if alert_count <= maximum_alerts:
            cutoff = score
            selected_alerts = alert_count
    return {
        "score_cutoff": cutoff,
        "training_control_hours": hours,
        "training_control_alerts": selected_alerts,
        "training_control_alert_limit": maximum_alerts,
        "budget_per_1000_control_hours": budget_per_1000h,
    }


def replay_validation(
    rows: list[dict[str, Any]], indices: list[int], scores: np.ndarray, cutoff: float,
) -> dict[str, Any]:
    if len(indices) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Validation scores must be finite and align with rows")
    events: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    hours = 0.0
    for index, score in zip(indices, scores):
        row = rows[index]
        groups[row["group_id"]].append((index, float(score)))
        if row["source_cohort"] == "control_sample":
            hours += row["airborne_exposure_seconds"] / 3600.0
        for event in row["future_events"]:
            events.setdefault(event["event_id"], {
                "event_id": event["event_id"],
                "date_utc": row["date_utc"],
                "icao24": row["icao24"],
                "event_unix_s": float(event["event_unix_s"]),
                "signal_subtypes": event["signal_subtypes"],
            })
    detected: dict[str, dict[str, Any]] = {}
    false_alerts = 0
    candidate_negative_alerts = 0
    false_alert_examples: list[dict[str, Any]] = []
    candidate_negative_examples: list[dict[str, Any]] = []
    for examples in groups.values():
        last_alert = -math.inf
        for index, score in sorted(examples, key=lambda item: rows[item[0]]["anchor_unix_s"]):
            row = rows[index]
            anchor = row["anchor_unix_s"]
            if score < cutoff or anchor - last_alert < SUPPRESSION_SECONDS:
                continue
            last_alert = anchor
            linked = {event["event_id"] for event in row["future_events"]}
            if linked:
                for event_id in linked:
                    if event_id not in detected:
                        detected[event_id] = {
                            **events[event_id],
                            "warning_seconds": events[event_id]["event_unix_s"] - anchor,
                            "score": score,
                        }
            elif row["source_cohort"] == "control_sample":
                false_alerts += 1
                false_alert_examples.append({
                    "example_id": row["example_id"], "date_utc": row["date_utc"],
                    "icao24": row["icao24"], "anchor_unix_s": anchor, "score": score,
                })
            else:
                candidate_negative_alerts += 1
                candidate_negative_examples.append({
                    "example_id": row["example_id"], "date_utc": row["date_utc"],
                    "icao24": row["icao24"], "anchor_unix_s": anchor, "score": score,
                })
    found = sorted(detected.values(), key=lambda item: (item["date_utc"], item["icao24"], item["event_unix_s"]))
    missed = sorted((event for event_id, event in events.items() if event_id not in detected),
                    key=lambda item: (item["date_utc"], item["icao24"], item["event_unix_s"]))
    warnings = [item["warning_seconds"] for item in found]
    return {
        "event_count": len(events),
        "detected_events": len(found),
        "detected_aircraft": len({item["icao24"] for item in found}),
        "event_recall": len(found) / len(events) if events else None,
        "warning_time_median_seconds": float(np.median(warnings)) if warnings else None,
        "control_hours": hours,
        "control_false_alerts": false_alerts,
        "control_false_alerts_per_1000_hours": false_alerts * 1000.0 / hours if hours else None,
        "candidate_negative_alerts": candidate_negative_alerts,
        "control_false_alert_examples": false_alert_examples,
        "candidate_negative_alert_examples": candidate_negative_examples,
        "matched_events": found,
        "missed_events": missed,
    }


def score_diagnostics(
    rows: list[dict[str, Any]], indices: list[int], scores: np.ndarray, cutoff: float,
) -> dict[str, Any]:
    labels = np.asarray([rows[index]["label"] for index in indices], dtype=int)
    controls = np.asarray([rows[index]["source_cohort"] == "control_sample" for index in indices])
    above = scores >= cutoff
    event_best: dict[str, dict[str, Any]] = {}
    for index, score in zip(indices, scores):
        row = rows[index]
        for event in row["future_events"]:
            event_id = event["event_id"]
            if event_id not in event_best or score > event_best[event_id]["best_pre_event_score"]:
                event_best[event_id] = {
                    "event_id": event_id,
                    "icao24": row["icao24"],
                    "signal_subtypes": event["signal_subtypes"],
                    "best_pre_event_score": float(score),
                    "best_warning_seconds": float(event["event_unix_s"]) - row["anchor_unix_s"],
                }
    high_controls = sorted(
        ({"example_id": rows[index]["example_id"], "icao24": rows[index]["icao24"],
          "anchor_unix_s": rows[index]["anchor_unix_s"], "score": float(score)}
         for index, score, is_control in zip(indices, scores, controls) if is_control),
        key=lambda item: item["score"], reverse=True,
    )[:10]
    return {
        "window_roc_auc": float(roc_auc_score(labels, scores)),
        "window_average_precision": float(average_precision_score(labels, scores)),
        "positive_window_fraction": float(np.mean(labels)),
        "positive_windows_above_cutoff": int(np.sum((labels == 1) & above)),
        "events_with_any_window_above_cutoff_before_suppression": sum(
            item["best_pre_event_score"] >= cutoff for item in event_best.values()),
        "median_positive_score": float(np.median(scores[labels == 1])),
        "median_control_score": float(np.median(scores[controls])),
        "highest_scoring_control_windows": high_controls,
        "event_best_pre_declaration_scores": sorted(
            event_best.values(), key=lambda item: item["best_pre_event_score"], reverse=True),
    }


def existing_baseline_results(dataset: Path, date: str) -> dict[str, Any]:
    path = dataset.parent / "temporal_episode_comparison" / "temporal_model_comparison.json"
    if not path.is_file():
        return {"available": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {"available": True, "source_report": str(path.resolve())}
    for name in ("isolation_forest", "temporal_boosted_tree_baseline"):
        day = next(item for item in report[name]["per_date_results"] if item["date_utc"] == date)
        eligible = [point for point in day["alert_curve"]
                    if point["control_false_alerts_per_1000_hours"] is not None
                    and point["control_false_alerts_per_1000_hours"] <= FALSE_ALERT_BUDGET_PER_1000H]
        best = max(eligible, key=lambda point: (
            point["detected_events"], -point["control_false_alerts_per_1000_hours"],
        ))
        results[name] = {
            "events": day["events"], "detected_events_at_or_below_25_false_alerts_per_1000h": best["detected_events"],
            "window_roc_auc": day["roc_auc"], "window_average_precision": day["average_precision"],
        }
    return results


def run_pilot(dataset: Path, output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, rows = load_compact_dataset(dataset, output_dir / "compact_episode_cache.joblib")
    training, validation = pilot_split(rows)
    selected = sample_training_windows(rows, training)
    weights = event_balanced_weights(rows, selected)
    labels = np.asarray([rows[index]["label"] for index in selected], dtype=int)
    train_controls = [index for index in training if rows[index]["source_cohort"] == "control_sample"]
    validation_events = {event["event_id"] for index in validation for event in rows[index]["future_events"]}
    if len(validation_events) != 38:
        raise ValueError(f"Pilot date expected 38 eligible events, found {len(validation_events)}")
    print(f"Pilot: {len(selected)} training windows, {sum(labels)} positive windows, "
          f"{len(validation)} validation windows, {len(validation_events)} events", flush=True)

    experiments = []
    passing_model = None
    for name in MODEL_CONFIGS:
        model_started = time.monotonic()
        print(f"Fitting {name} once", flush=True)
        model = build_model(name)
        model.fit(matrix[selected], labels, sample_weight=weights)
        training_auc = float(roc_auc_score(labels, model.predict_proba(matrix[selected])[:, 1]))
        control_scores = model.predict_proba(matrix[train_controls])[:, 1]
        cutoff = cutoff_from_training_controls(rows, train_controls, control_scores)
        validation_scores = model.predict_proba(matrix[validation])[:, 1]
        replay = replay_validation(rows, validation, validation_scores, float(cutoff["score_cutoff"]))
        diagnostics = score_diagnostics(rows, validation, validation_scores, float(cutoff["score_cutoff"]))
        event_scores = diagnostics["event_best_pre_declaration_scores"]
        if len(event_scores) >= 3:
            third_score = event_scores[2]["best_pre_event_score"]
            diagnostic_replay = replay_validation(rows, validation, validation_scores, third_score)
            diagnostics["posthoc_third_event_score_replay"] = {
                "score_cutoff": third_score,
                "detected_events_after_suppression": diagnostic_replay["detected_events"],
                "control_false_alerts": diagnostic_replay["control_false_alerts"],
                "control_false_alerts_per_1000_hours": diagnostic_replay["control_false_alerts_per_1000_hours"],
                "note": "Diagnostic use of held-out event scores; this cutoff is not selected or deployed.",
            }
        maximum_validation_alerts = math.floor(FALSE_ALERT_BUDGET_PER_1000H * replay["control_hours"] / 1000.0)
        passed = (replay["detected_events"] >= 3
                  and replay["detected_aircraft"] >= 2
                  and replay["control_false_alerts"] <= maximum_validation_alerts)
        experiments.append({
            "model": name,
            "configuration": MODEL_CONFIGS[name],
            "training_seconds": time.monotonic() - model_started,
            "sampled_training_window_roc_auc": training_auc,
            "experimental_cutoff_from_training_controls": cutoff,
            "validation_false_alert_limit": maximum_validation_alerts,
            "passed_pilot_gate": passed,
            "validation": replay,
            "score_diagnostics": diagnostics,
        })
        print(f"{name}: detected={replay['detected_events']}/{replay['event_count']} "
              f"aircraft={replay['detected_aircraft']} control_false_alerts="
              f"{replay['control_false_alerts']}/{maximum_validation_alerts} passed={passed}", flush=True)
        if passed:
            passing_model = name
            break

    report = {
        "schema_version": 1,
        "target": EPISODE_TARGET,
        "pilot_validation_date": PILOT_DATE,
        "pilot_only": True,
        "passed_pilot_gate": passing_model is not None,
        "passing_model": passing_model,
        "training_dates": sorted({rows[index]["date_utc"] for index in training}),
        "training_windows_after_sampling": len(selected),
        "training_positive_windows": int(sum(labels)),
        "training_distinct_events": len({event["event_id"] for index in selected
                                         for event in rows[index]["future_events"]}),
        "training_negative_windows": int(len(selected) - sum(labels)),
        "validation_windows": len(validation),
        "weighting": "equal positive mass per real event and equal negative mass per aircraft-day/cohort; 1:1 class mass",
        "negative_windows_per_aircraft_day_cohort": NEGATIVE_WINDOWS_PER_AIRCRAFT_DAY,
        "gate": {
            "minimum_detected_events": 3,
            "minimum_detected_aircraft": 2,
            "maximum_false_alerts_per_1000_control_hours": FALSE_ALERT_BUDGET_PER_1000H,
            "suppression_seconds": SUPPRESSION_SECONDS,
        },
        "experiments": experiments,
        "existing_baselines_on_pilot_date": existing_baseline_results(dataset, PILOT_DATE),
        "elapsed_seconds": time.monotonic() - started,
        "warning": "Development-only pilot. No ten-fold training or operational threshold is implied by this report.",
    }
    path = output_dir / f"pilot_{PILOT_DATE}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pilot report: {path.resolve()}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/learning/ten_day_development/declaration_episode_windows.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/learning/ten_day_development/event_supervised"))
    args = parser.parse_args()
    run_pilot(args.dataset, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
