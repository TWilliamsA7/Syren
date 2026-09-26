"""Inspect declaration-proxy signal and train-versus-date fit without model sweeps."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.compare_temporal_models import (
    SEED, TREE_CONFIG, fit_isolation_forest, isolation_anomaly_scores,
    load_rows, purged_date_split,
)
from ml.compare_models import fit_predict
from ml.features import TEMPORAL_FEATURE_NAMES
from ml.sequence_features import CHANNEL_NAMES


def _key(row: dict[str, Any]) -> str:
    return f"{row['date_utc']}:{row['icao24']}:{row['event_unix_s']}"


def _add(total: np.ndarray, value: np.ndarray) -> np.ndarray:
    return total + value


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "p10": None, "median": None, "p90": None}
    return {
        "n": len(values),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _prediction_diagnostics(prediction_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in ("temporal_boosted_tree_baseline", "isolation_forest"):
        path = prediction_dir / f"{model}_oof_predictions.jsonl"
        event_scores: dict[str, list[float]] = defaultdict(list)
        control_scores: dict[str, list[float]] = defaultdict(list)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                score = float(row["score"])
                if row["label"]:
                    for event in row.get("future_events", []):
                        event_id = event["event_id"]
                        event_scores[event_id].append(score)
                elif row["source_cohort"] == "control_sample":
                    control_scores[row["group_id"]].append(score)
        event_means = [float(np.mean(values)) for values in event_scores.values()]
        control_means = [float(np.mean(values)) for values in control_scores.values()]
        event_max = [float(max(values)) for values in event_scores.values()]
        control_max = [float(max(values)) for values in control_scores.values()]
        result[model] = {
            "event_mean_score_vs_control_group_mean_auc": _auc(event_means, control_means),
            "event_mean_score": _distribution(event_means),
            "control_group_mean_score": _distribution(control_means),
            "event_max_score_vs_control_group_max_auc_opportunity_biased": _auc(event_max, control_max),
            "event_max_score": _distribution(event_max),
            "control_group_max_score": _distribution(control_max),
            "caveat": (
                "Event score means aggregate positive windows within each event; control means aggregate all "
                "windows within each sampled control flight segment. The max-score comparison is "
                "opportunity-biased because control segments often have more windows. These are "
                "descriptive diagnostics, not alert operating metrics."
            ),
        }
    return result


def diagnose(dataset: Path, prediction_dir: Path, fit_gap_date: str, output: Path) -> dict[str, Any]:
    width = len(TEMPORAL_FEATURE_NAMES)
    event_sum: dict[str, np.ndarray] = {}
    event_count: Counter[str] = Counter()
    event_subtypes: dict[str, set[str]] = defaultdict(set)
    group_sum: dict[str, np.ndarray] = {}
    group_count: Counter[str] = Counter()
    group_label: dict[str, str] = {}
    mask_sum: dict[tuple[str, str], np.ndarray] = {}
    mask_count: Counter[tuple[str, str]] = Counter()
    window_mask_sum: Counter[str] = Counter()
    window_mask_count: Counter[str] = Counter()
    window_channel_sum: dict[str, np.ndarray] = {}

    with dataset.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            features = np.fromiter(
                (float(row["temporal_features"][name]) for name in TEMPORAL_FEATURE_NAMES),
                dtype=np.float64, count=width,
            )
            sequence = np.asarray(row["sequence"], dtype=np.float32)
            masks = sequence[:, 6:11].mean(axis=0).astype(float).tolist()
            label = int(row["label"])
            source = row["source_cohort"]
            if label:
                event_id = _key(row)
                event_sum[event_id] = _add(event_sum.get(event_id, np.zeros(width)), features)
                event_count[event_id] += 1
                event_subtypes[event_id].update(row.get("event_signal_subtypes", []))
                key = ("event_positive", event_id)
                mask_sum[key] = _add(mask_sum.get(key, np.zeros(5)), np.asarray(masks))
                mask_count[key] += 1
                window_mask_sum["event_positive"] += float(np.mean(masks))
                window_mask_count["event_positive"] += 1
                window_channel_sum["event_positive"] = _add(
                    window_channel_sum.get("event_positive", np.zeros(5)), np.asarray(masks))
            else:
                group_id = row["group_id"]
                group_sum[group_id] = _add(group_sum.get(group_id, np.zeros(width)), features)
                group_count[group_id] += 1
                group_label[group_id] = "control_negative" if source == "control_sample" else "candidate_negative"
                key = (group_label[group_id], group_id)
                mask_sum[key] = _add(mask_sum.get(key, np.zeros(5)), np.asarray(masks))
                mask_count[key] += 1
                window_mask_sum[group_label[group_id]] += float(np.mean(masks))
                window_mask_count[group_label[group_id]] += 1
                window_channel_sum[group_label[group_id]] = _add(
                    window_channel_sum.get(group_label[group_id], np.zeros(5)), np.asarray(masks))
            if line_number % 25000 == 0:
                print(f"Aggregated {line_number:,} windows", flush=True)

    events = {key: total / event_count[key] for key, total in event_sum.items()}
    groups = {key: total / group_count[key] for key, total in group_sum.items()}
    control_ids = [key for key in groups if group_label[key] == "control_negative"]
    candidate_ids = [key for key in groups if group_label[key] == "candidate_negative"]
    event_values = list(events.values())

    feature_report = []
    for index, name in enumerate(TEMPORAL_FEATURE_NAMES):
        positive = [float(values[index]) for values in event_values]
        control = [float(groups[key][index]) for key in control_ids]
        candidate = [float(groups[key][index]) for key in candidate_ids]
        feature_report.append({
            "feature": name,
            "positive_event_mean": float(np.mean(positive)),
            "positive_event_median": float(np.median(positive)),
            "control_segment_mean": float(np.mean(control)) if control else None,
            "control_segment_median": float(np.median(control)) if control else None,
            "event_vs_control_group_auc": _auc(positive, control),
            "candidate_negative_group_mean": float(np.mean(candidate)) if candidate else None,
            "event_vs_candidate_negative_group_auc": _auc(positive, candidate),
        })
    feature_report.sort(key=lambda item: abs((item["event_vs_control_group_auc"] or 0.5) - 0.5), reverse=True)

    subtype_counts = Counter(subtype for subtypes in event_subtypes.values() for subtype in subtypes)
    mask_report = {}
    for cohort in ("event_positive", "control_negative", "candidate_negative"):
        keys = [key for key in mask_sum if key[0] == cohort]
        summary_matrix = np.asarray([mask_sum[key] / mask_count[key] for key in keys], dtype=float)
        mask_report[cohort] = {
            "windows": int(window_mask_count[cohort]),
            "events_or_segments": len(keys),
            "window_mean_valid_fraction_overall": (
                float(window_mask_sum[cohort] / window_mask_count[cohort])
                if window_mask_count[cohort] else None
            ),
            "window_mean_valid_fraction_by_channel": {
                name: float(window_channel_sum[cohort][i] / window_mask_count[cohort])
                if window_mask_count[cohort] else None
                for i, name in enumerate(CHANNEL_NAMES[6:])
            },
            "event_or_segment_mean_valid_fraction_by_channel": {
                name: float(summary_matrix[:, i].mean()) if len(summary_matrix) else None
                for i, name in enumerate(CHANNEL_NAMES[6:])
            },
        }

    prediction_diagnostics = _prediction_diagnostics(prediction_dir)
    event_count_total = len(events)
    control_count_total = len(control_ids)
    candidate_count_total = len(candidate_ids)
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Descriptive signal and fit diagnostics for the ten-day ADS-B declaration-proxy dataset.",
        "dataset": str(dataset.resolve()),
        "prediction_dir": str(prediction_dir.resolve()),
        "event_count": event_count_total,
        "negative_segments": {
            "sampled_controls": control_count_total,
            "candidate_aircraft": candidate_count_total,
        },
        "event_subtype_counts_overlapping": dict(sorted(subtype_counts.items())),
        "sequence_mask_coverage": mask_report,
        "top_univariate_feature_separation": feature_report[:15],
        "all_univariate_feature_separation": feature_report,
        "out_of_fold_score_group_diagnostics": prediction_diagnostics,
        "limitations": [
            "An ADS-B declaration is an operational/reporting action, not a directly observed onset of danger.",
            "Feature AUCs compare one averaged representation per positive event with one averaged representation per flight segment; they are descriptive and not independent-event uncertainty estimates.",
            "Candidate-negative segments and sampled-control segments have different selection mechanisms.",
            "All ten dates were used in model development; these diagnostics do not provide a final test estimate.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    del event_sum, event_count, event_subtypes, group_sum, group_count
    del group_label, mask_sum, mask_count, window_mask_sum, window_mask_count
    del window_channel_sum, events, groups, control_ids, candidate_ids, event_values
    gc.collect()

    # One fixed date fold quantifies fit on the training rows versus a genuinely
    # date-held-out sample; this is a diagnostic, not a new configuration search.
    rows = load_rows(dataset)
    train, validation, purged = purged_date_split(rows, fit_gap_date)
    train_y = [int(row["label"]) for row in train]
    val_y = [int(row["label"]) for row in validation]
    print(f"Fit-gap fold {fit_gap_date}: fitting fixed temporal boosted tree", flush=True)
    tree_train, tree_val = fit_predict(train, validation, TREE_CONFIG)
    print(f"Fit-gap fold {fit_gap_date}: fitting control-only Isolation Forest", flush=True)
    forest, _ = fit_isolation_forest(
        train, SEED + sorted({r["date_utc"] for r in rows}).index(fit_gap_date),
    )
    forest_train = isolation_anomaly_scores(forest, train)
    forest_val = isolation_anomaly_scores(forest, validation)

    def fit_metrics(y: list[int], scores: np.ndarray) -> dict[str, float]:
        return {
            "window_roc_auc": float(roc_auc_score(y, scores)),
            "window_average_precision": float(average_precision_score(y, scores)),
        }

    fit_gap = {
        "validation_date": fit_gap_date,
        "purged_aircraft_ids": len(purged),
        "train_windows": len(train),
        "validation_windows": len(validation),
        "train_positive_windows": sum(train_y),
        "validation_positive_windows": sum(val_y),
        "train_positive_fraction_random_rank_ap_baseline": float(np.mean(train_y)),
        "validation_positive_fraction_random_rank_ap_baseline": float(np.mean(val_y)),
        "temporal_hgb": {"train_in_sample": fit_metrics(train_y, tree_train),
                         "validation": fit_metrics(val_y, tree_val)},
        "isolation_forest": {"train_in_sample": fit_metrics(train_y, forest_train),
                             "validation": fit_metrics(val_y, forest_val)},
        "interpretation": (
            "Large training-to-validation gap indicates overfit or date/label shift, not underfitting. "
            "Low scores on both sides indicate weak input signal, a mismatched proxy target, or both."
        ),
    }

    report["single_fold_fit_gap"] = fit_gap
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/learning/ten_day_development/declaration_episode_windows.jsonl"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("data/learning/ten_day_development/temporal_episode_comparison"))
    parser.add_argument("--fit-gap-date", default="2025-08-15")
    parser.add_argument("--output", type=Path, default=Path("data/learning/ten_day_development/temporal_signal_diagnostics.json"))
    args = parser.parse_args()
    report = diagnose(args.dataset, args.prediction_dir, args.fit_gap_date, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "event_count": report["event_count"],
        "top_feature_separation": report["top_univariate_feature_separation"][:5],
        "single_fold_fit_gap": report["single_fold_fit_gap"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
