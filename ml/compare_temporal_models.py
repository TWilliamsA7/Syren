"""Evaluate a control-only Isolation Forest against the fixed temporal HGB baseline."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import scipy
import sklearn
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.compare_models import fit_predict as fit_tree
from ml.features import TEMPORAL_FEATURE_NAMES
from ml.label_windows import DEFAULT_OUTPUT, EPISODE_TARGET

SEED = 17
TREE_CONFIG = {
    "features": "temporal", "weighting": "group_balanced",
    "model": "hist_gradient_boosting", "max_leaf_nodes": 3,
    "max_iter": 100,
}
ISOLATION_FOREST_CONFIG = {
    "model": "isolation_forest",
    "training_cohort": "control_sample only",
    "max_windows_per_aircraft_day": 4,
    "n_estimators": 200,
    "max_samples": 256,
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
}
CURVE_POINTS = 201


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target") != EPISODE_TARGET:
                raise ValueError(f"Target mismatch at {path}:{line_no}")
            if not isinstance(row.get("targets"), dict) or not isinstance(row.get("future_events"), list):
                raise ValueError(f"Missing multi-label targets or episode metadata at {path}:{line_no}")
            # The primary tabular predictor and its tree reference use only
            # temporal_features; avoid retaining the much larger sequence field.
            row.pop("sequence", None)
            rows.append(row)
    if not rows:
        raise ValueError("No labeled rows found")
    return rows


def purged_date_split(rows: list[dict[str, Any]], validation_date: str) -> tuple[list[dict], list[dict], set[str]]:
    validation = [row for row in rows if row["date_utc"] == validation_date]
    if not validation:
        raise ValueError(f"No rows for validation date {validation_date}")
    validation_aircraft = {row["icao24"] for row in validation}
    train = [row for row in rows if row["date_utc"] != validation_date
             and row["icao24"] not in validation_aircraft]
    remaining_overlap = {row["icao24"] for row in train} & validation_aircraft
    if remaining_overlap:
        raise AssertionError("Aircraft purge failed")
    all_other_aircraft = {row["icao24"] for row in rows if row["date_utc"] != validation_date}
    return train, validation, all_other_aircraft & validation_aircraft


def target_labels(rows: list[dict[str, Any]], head: str) -> np.ndarray:
    return np.asarray([int(row["targets"][head]) for row in rows], dtype=np.float32)


def temporal_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = []
    for index, row in enumerate(rows):
        features = row.get("temporal_features")
        if not isinstance(features, dict) or tuple(features) != TEMPORAL_FEATURE_NAMES:
            raise ValueError(f"Temporal feature schema mismatch at row {index}")
        values = []
        for name in TEMPORAL_FEATURE_NAMES:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Feature {name} must be finite numeric at row {index}")
            values.append(float(value))
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def balanced_control_sample(
    rows: list[dict[str, Any]], max_windows_per_aircraft_day: int = 4,
) -> list[dict[str, Any]]:
    """Keep up to four evenly spaced negative control windows per aircraft-day."""
    if max_windows_per_aircraft_day < 1:
        raise ValueError("max_windows_per_aircraft_day must be positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("source_cohort") != "control_sample":
            raise ValueError("Isolation Forest may train only on control_sample rows")
        if int(row.get("targets", {}).get("any_declaration", -1)) != 0:
            raise ValueError("Control rows must have no future declaration label")
        key = (str(row["date_utc"]), str(row["icao24"]))
        groups.setdefault(key, []).append(row)

    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: float(row["anchor_unix_s"]))
        if len(group) > max_windows_per_aircraft_day:
            indices = np.linspace(
                0, len(group) - 1, max_windows_per_aircraft_day, dtype=int,
            )
            group = [group[index] for index in indices]
        selected.extend(group)
    if len(selected) < 2:
        raise ValueError("Isolation Forest needs at least two sampled control windows")
    return selected


def fit_isolation_forest(
    training_rows: list[dict[str, Any]], seed: int = SEED,
) -> tuple[IsolationForest, list[dict[str, Any]]]:
    controls = [row for row in training_rows if row.get("source_cohort") == "control_sample"]
    sampled = balanced_control_sample(controls, ISOLATION_FOREST_CONFIG["max_windows_per_aircraft_day"])
    model = IsolationForest(
        n_estimators=ISOLATION_FOREST_CONFIG["n_estimators"],
        max_samples=min(ISOLATION_FOREST_CONFIG["max_samples"], len(sampled)),
        contamination=ISOLATION_FOREST_CONFIG["contamination"],
        max_features=ISOLATION_FOREST_CONFIG["max_features"],
        bootstrap=ISOLATION_FOREST_CONFIG["bootstrap"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(temporal_feature_matrix(sampled))
    return model, sampled


def isolation_anomaly_scores(model: IsolationForest, rows: list[dict[str, Any]]) -> np.ndarray:
    """Return larger scores for more anomalous windows; these are not probabilities."""
    if not rows:
        return np.asarray([], dtype=float)
    return -model.decision_function(temporal_feature_matrix(rows))


def wilson(successes: int, trials: int) -> list[float] | None:
    if not trials:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - half, center + half]


def poisson_rate_interval(count: int, hours: float) -> list[float] | None:
    if hours <= 0:
        return None
    lower = 0.5 * chi2.ppf(0.025, 2 * count) if count else 0.0
    upper = 0.5 * chi2.ppf(0.975, 2 * (count + 1))
    return [float(lower * 1000 / hours), float(upper * 1000 / hours)]


def curve_thresholds(scores: list[float]) -> list[float]:
    # Cover the model's full score range (Isolation Forest scores are not bounded
    # to [0, 1]) and include a distinct no-alert point.
    if not scores:
        return [0.0]
    low, high = min(scores), max(scores)
    grid = np.linspace(low, high, CURVE_POINTS).tolist() if low != high else [low]
    return sorted(set(grid + [math.nextafter(high, math.inf)]))


def episode_catalog(rows: list[dict[str, Any]], head: str = "any_declaration") -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for row in rows:
        for event in row.get("future_events", []):
            subtypes = sorted(set(event.get("signal_subtypes", [])))
            if head != "any_declaration" and head not in subtypes:
                continue
            events.setdefault(event["event_id"], {
                "event_unix_s": float(event["event_unix_s"]),
                "date_utc": row["date_utc"], "icao24": row["icao24"],
                "signal_subtypes": subtypes,
            })
    return events


def replay_episodes(
    rows: list[dict[str, Any]], scores: list[float], threshold: float,
    head: str = "any_declaration", suppression_seconds: float = 600,
) -> dict[str, Any]:
    if len(rows) != len(scores):
        raise ValueError("Rows and scores have different lengths")
    events = episode_catalog(rows, head)
    exposure = sum(float(row.get("airborne_exposure_seconds", 0.0))
                   for row in rows if row.get("source_cohort") == "control_sample") / 3600
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row["group_id"], []).append((row, float(score)))
    detected: dict[str, float] = {}
    false_alerts = 0
    for examples in grouped.values():
        last_alert = -math.inf
        for row, score in sorted(examples, key=lambda pair: float(pair[0]["anchor_unix_s"])):
            anchor = float(row["anchor_unix_s"])
            if score < threshold or anchor - last_alert < suppression_seconds:
                continue
            last_alert = anchor
            matched = []
            for event in row.get("future_events", []):
                event_id = event["event_id"]
                if event_id in events and (head == "any_declaration" or head in event.get("signal_subtypes", [])):
                    matched.append(event_id)
            if matched:
                for event_id in matched:
                    detected.setdefault(event_id, float(events[event_id]["event_unix_s"]) - anchor)
            elif row.get("source_cohort") == "control_sample":
                false_alerts += 1
    warnings = sorted(detected.values())
    return {
        "threshold": float(threshold), "suppression_seconds": suppression_seconds,
        "event_count": len(events), "detected_events": len(detected),
        "event_recall": len(detected) / len(events) if events else None,
        "warning_seconds": warnings,
        "control_observed_airborne_hours": exposure,
        "control_false_alerts": false_alerts,
        "control_false_alerts_per_1000_hours": false_alerts * 1000 / exposure if exposure else None,
    }


def alert_metrics(
    rows: list[dict[str, Any]], scores: list[float], threshold: float,
    head: str = "any_declaration",
) -> dict[str, Any]:
    result = replay_episodes(rows, scores, threshold, head)
    warning = result["warning_seconds"]
    result["warning_time_median_seconds"] = float(np.median(warning)) if warning else None
    result["warning_time_p10_seconds"] = float(np.percentile(warning, 10)) if warning else None
    result.pop("warning_seconds", None)
    result["event_recall_wilson_95"] = wilson(result["detected_events"], result["event_count"])
    result["false_alert_rate_poisson_95_per_1000h"] = poisson_rate_interval(
        result["control_false_alerts"], result["control_observed_airborne_hours"])
    return result


def event_subtype_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for event in episode_catalog(rows).values():
        for name in event["signal_subtypes"] or ["unknown"]:
            totals[name] += 1
    return dict(sorted(totals.items()))


def audit_context(dataset: Path) -> dict[str, Any]:
    audit_path = dataset.parent / "candidate_audit.jsonl"
    summary_path = dataset.with_name(f"{dataset.stem}_summary.json")
    if not audit_path.is_file():
        return {"available": False}
    included = excluded = 0
    subtype_totals: Counter[str] = Counter()
    audits: dict[tuple[str, str], dict[str, Any]] = {}
    usable_statuses = {"general", "minfuel", "nordo", "unlawful", "downed"}
    usable_squawks = {"7700", "7600", "7500"}
    with audit_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            audits[(item["date_utc"], str(item["icao24"]).lower())] = item
            if item["decision"] == "include_proxy":
                included += 1
                for signal in item.get("signals", []):
                    name = str(signal.get("value", "")).lower()
                    kind = signal.get("signal")
                    if (kind == "emergency_field" and name in usable_statuses) or (
                        kind == "emergency_squawk" and name in usable_squawks
                    ):
                        subtype_totals[f"{kind}:{name}"] += 1
            elif item["decision"] == "exclude_proxy":
                excluded += 1
    label_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    no_window_reasons: Counter[str] = Counter()
    for value in label_summary.get("candidates_without_positive_windows", []):
        day, icao = value.split(":", 1)
        item = audits.get((day, icao.lower()), {})
        legs = item.get("signal_leg_coverage", [])
        if not legs:
            reason = "declaration_outside_observed_flight_leg"
        else:
            elapsed = [
                float(item["usable_onset_unix_s"])
                - datetime.fromisoformat(leg["start_utc"].replace("Z", "+00:00")).timestamp()
                for leg in legs
            ]
            if all(int(leg["points"]) < 10 for leg in legs):
                reason = "matching_leg_has_fewer_than_10_points"
            elif max(elapsed, default=0.0) < 420.0:
                reason = "less_than_420_seconds_of_leg_history_before_declaration"
            else:
                reason = "other_window_eligibility_or_stride_gap"
        no_window_reasons[reason] += 1
    return {
        "available": True,
        "candidate_traces_included_as_declaration_proxies": included,
        "candidate_traces_excluded_by_proxy_policy": excluded,
        "audited_usable_signal_counts_including_traces_without_eligible_windows": dict(sorted(subtype_totals.items())),
        "candidate_traces_without_positive_windows": len(label_summary.get("candidates_without_positive_windows", [])),
        "no_positive_window_reasons": dict(sorted(no_window_reasons.items())),
    }


def write_curve_svg(report: dict[str, Any], path: Path) -> None:
    """Write a dependency-free pooled event-recall/false-alert plot."""
    series = (
        ("isolation_forest", "Isolation Forest anomaly score", "#c026d3"),
        ("temporal_boosted_tree_baseline", "Temporal boosted tree", "#2563eb"),
    )
    curves = [report[name]["recall_vs_false_alert_curve"] for name, _, _ in series]
    xmax = max(float(point["control_false_alerts_per_1000_hours"] or 0.0)
               for curve in curves for point in curve)
    xmax = max(1.0, xmax)
    width, height = 1000, 620
    left, right, top, bottom = 100, 35, 90, 90
    plot_width, plot_height = width - left - right, height - top - bottom

    def x_position(rate: float) -> float:
        return left + math.log1p(max(0.0, rate)) / math.log1p(xmax) * plot_width

    def y_position(recall: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, recall))) * plot_height

    ticks = [0, 1, 5, 10, 25, 50, 100, 500, 1000, 5000, xmax]
    ticks = sorted(set(value for value in ticks if value <= xmax) | {xmax})
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="100" y="34" font-family="Segoe UI,Arial,sans-serif" font-size="23" font-weight="600" fill="#111827">Isolation Forest vs temporal baseline</text>',
        '<text x="100" y="60" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="#4b5563">Ten-date out-of-fold development comparison · ADS-B declaration proxy · no threshold selected</text>',
    ]
    for tick in ticks:
        x = x_position(tick)
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#4b5563">{tick:g}</text>')
    for recall_tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_position(recall_tick)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#4b5563">{recall_tick:.1f}</text>')
    elements.extend([
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 25}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="#111827">False alerts per 1,000 observed control airborne hours (log scale)</text>',
        f'<text transform="translate(24 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="#111827">Event recall</text>',
    ])
    legend_x = left + 25
    for index, ((key, label, color), curve) in enumerate(zip(series, curves)):
        points = []
        # Descending score threshold traces the complete sweep from no alerts to all alerts.
        for item in reversed(curve):
            rate = item["control_false_alerts_per_1000_hours"]
            recall = item["event_recall"]
            if rate is not None and recall is not None:
                points.append(f'{x_position(float(rate)):.1f},{y_position(float(recall)):.1f}')
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        legend_y = top + 18 + index * 26
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        auc = report[key]["oof_roc_auc_secondary"]
        elements.append(f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#111827">{label} (secondary ROC AUC {auc:.3f})</text>')
    elements.append('<text x="100" y="600" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#6b7280">Nominal 95% uncertainty intervals are in the JSON report; all dates are development data.</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def summarize_curves(
    rows: list[dict[str, Any]], scores: list[float], head: str = "any_declaration",
) -> tuple[list[dict], list[dict]]:
    thresholds = curve_thresholds(scores)
    curve = []
    for number, threshold in enumerate(thresholds, 1):
        result = alert_metrics(rows, scores, threshold, head)
        curve.append(result)
        if number % 50 == 0:
            print(f"  alert curve {number}/{len(thresholds)}", flush=True)
    # Per-date results use the same score-domain grid and keep dates visible.
    per_date = []
    for day in sorted({row["date_utc"] for row in rows}):
        selected = [(row, score) for row, score in zip(rows, scores) if row["date_utc"] == day]
        date_rows = [item[0] for item in selected]
        date_scores = [float(item[1]) for item in selected]
        y = [int(row["targets"][head]) for row in date_rows]
        per_date_curve = [alert_metrics(date_rows, date_scores, threshold, head) for threshold in thresholds]
        date_events = episode_catalog(date_rows, head)
        per_date.append({
            "date_utc": day,
            "windows": len(date_rows),
            "positive_windows": sum(y),
            "events": len(date_events),
            "aircraft_days": len({(row["date_utc"], row["icao24"]) for row in date_rows}),
            "roc_auc": float(roc_auc_score(y, date_scores)) if len(set(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, date_scores)) if sum(y) else None,
            "event_signal_subtype_counts": event_subtype_counts(date_rows),
            "alert_curve": per_date_curve,
        })
    return curve, per_date


def best_recall_at_false_alert_limit(curve: list[dict[str, Any]], limit: float) -> dict[str, Any]:
    eligible = [point for point in curve
                if point["control_false_alerts_per_1000_hours"] is not None
                and point["control_false_alerts_per_1000_hours"] <= limit]
    if not eligible:
        return {"false_alert_limit_per_1000_hours": limit, "event_recall": None}
    best = max(eligible, key=lambda point: (
        float(point["event_recall"] or 0.0),
        -float(point["control_false_alerts_per_1000_hours"] or 0.0),
    ))
    return {
        "false_alert_limit_per_1000_hours": limit,
        "event_recall": best["event_recall"],
        "false_alerts_per_1000_hours": best["control_false_alerts_per_1000_hours"],
        "detected_events": best["detected_events"],
        "event_count": best["event_count"],
        "warning_time_median_seconds": best["warning_time_median_seconds"],
        "warning_time_p10_seconds": best["warning_time_p10_seconds"],
    }


def describe_tradeoff_at_limit(
    isolation_forest: dict[str, Any], baseline: dict[str, Any],
) -> dict[str, Any]:
    limit = float(isolation_forest["false_alert_limit_per_1000_hours"])
    alert_word = "alert" if limit == 1 else "alerts"
    forest_recall = isolation_forest.get("event_recall")
    baseline_recall = baseline.get("event_recall")
    if forest_recall is None or baseline_recall is None:
        status = "not_comparable"
        summary = "Event recall could not be compared at this descriptive false-alert limit."
    elif forest_recall == baseline_recall == 0:
        status = "neither_detects_events"
        events = int(isolation_forest.get("event_count", 0))
        summary = (
            f"At the descriptive cap of no more than {limit:g} false {alert_word} per 1,000 control-flight hours, "
            f"neither model detected any of the {events} out-of-fold declaration episodes. "
            "Isolation Forest did not improve event recall at this budget. Secondary window-level ROC AUC and "
            "average precision are reported separately and do not establish event-level utility at this cap. "
            "All ten dates are development data."
        )
    elif forest_recall == baseline_recall:
        status = "equal_event_recall"
        summary = (
            f"At the descriptive cap of no more than {limit:g} false {alert_word} per 1,000 control-flight hours, "
            f"both models reached event recall {forest_recall:.4f}; Isolation Forest did not improve recall. "
            "All ten dates are development data."
        )
    elif forest_recall > baseline_recall:
        status = "isolation_forest_higher_recall"
        summary = (
            f"At the descriptive cap of no more than {limit:g} false {alert_word} per 1,000 control-flight hours, "
            f"Isolation Forest reached event recall {forest_recall:.4f} versus {baseline_recall:.4f} for the "
            "baseline. This is a development-only comparison across the ten available dates."
        )
    else:
        status = "baseline_higher_recall"
        summary = (
            f"At the descriptive cap of no more than {limit:g} false {alert_word} per 1,000 control-flight hours, "
            f"Isolation Forest reached event recall {forest_recall:.4f} versus {baseline_recall:.4f} for the "
            "baseline. This is a development-only comparison across the ten available dates."
        )
    return {
        "status": status,
        "false_alert_limit_per_1000_control_flight_hours": limit,
        "summary": summary,
    }


def load_cached_oof_scores(
    rows: list[dict[str, Any]], output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Load and validate complete OOF scores so final reporting can be resumed."""
    row_by_id = {row["example_id"]: row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("Dataset example_id values must be unique to reuse OOF predictions")
    score_types = {
        "isolation_forest": "isolation_anomaly_score",
        "temporal_boosted_tree_baseline": "ranking_score",
    }
    cached: dict[str, dict[str, float]] = {}
    for name, expected_type in score_types.items():
        path = output_dir / f"{name}_oof_predictions.jsonl"
        if not path.is_file():
            raise ValueError(f"Cannot reuse OOF predictions; missing {path}")
        values: dict[str, float] = {}
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                example_id = record.get("example_id")
                row = row_by_id.get(example_id)
                if row is None or example_id in values:
                    raise ValueError(f"Unexpected or duplicate example_id in {path}:{line_no}")
                score = record.get("score")
                if (record.get("score_type") != expected_type
                        or isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(score)):
                    raise ValueError(f"Invalid score or score type in {path}:{line_no}")
                if record.get("targets") != row.get("targets"):
                    raise ValueError(f"Cached OOF labels disagree with dataset at {path}:{line_no}")
                values[example_id] = float(score)
        if set(values) != set(row_by_id):
            raise ValueError(f"Cached OOF file {path} does not cover the dataset exactly once")
        cached[name] = values
    return cached


def compare(
    dataset: Path,
    output_dir: Path,
    development_dates: tuple[str, ...] | None = None,
    reuse_oof_predictions: bool = False,
) -> dict[str, Any]:
    rows = load_rows(dataset)
    observed_dates = sorted({row["date_utc"] for row in rows})
    dates = sorted(set(development_dates)) if development_dates else observed_dates
    if set(dates) != set(observed_dates) or len(dates) != 10:
        raise ValueError(f"Expected all ten development dates; observed={observed_dates}, requested={dates}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_scores = load_cached_oof_scores(rows, output_dir) if reuse_oof_predictions else None
    oof: dict[str, dict[str, Any]] = {
        "isolation_forest": {"rows": [], "scores": [], "folds": [], "score_type": "isolation_anomaly_score"},
        "temporal_boosted_tree_baseline": {"rows": [], "scores": [], "folds": [], "score_type": "ranking_score"},
    }
    fold_counts = []
    for fold_number, validation_date in enumerate(dates, 1):
        train, validation, purged = purged_date_split(rows, validation_date)
        if {row["label"] for row in train} != {0, 1}:
            raise ValueError(f"Training fold {validation_date} lacks both labels")
        seed = SEED + fold_number - 1
        print(
            f"Fold {fold_number}/10 {validation_date}: train={len(train)} "
            f"validation={len(validation)} purged_aircraft={len(purged)}", flush=True,
        )
        if cached_scores is None:
            _, tree_scores = fit_tree(train, validation, TREE_CONFIG)
            forest, normal_rows = fit_isolation_forest(train, seed)
            forest_scores = isolation_anomaly_scores(forest, validation)
        else:
            tree_scores = [cached_scores["temporal_boosted_tree_baseline"][row["example_id"]]
                           for row in validation]
            forest_scores = np.asarray([
                cached_scores["isolation_forest"][row["example_id"]] for row in validation
            ], dtype=float)
            normal_rows = balanced_control_sample([
                row for row in train if row.get("source_cohort") == "control_sample"
            ])
        y_val = target_labels(validation, "any_declaration").astype(int).tolist()
        validation_events = len(episode_catalog(validation))
        shared_fold = {
            "validation_date": validation_date,
            "train_dates": sorted({row["date_utc"] for row in train}),
            "train_aircraft_days": len({(row["date_utc"], row["icao24"]) for row in train}),
            "validation_aircraft_days": len({(row["date_utc"], row["icao24"]) for row in validation}),
            "purged_aircraft_ids": len(purged),
            "train_windows": len(train),
            "validation_windows": len(validation),
            "validation_positive_windows": sum(y_val),
            "validation_events": validation_events,
            "training_seed": seed,
        }
        for name, scores in (
            ("temporal_boosted_tree_baseline", np.asarray(tree_scores, dtype=float)),
            ("isolation_forest", forest_scores),
        ):
            score_list = [float(value) for value in scores]
            oof[name]["rows"].extend(validation)
            oof[name]["scores"].extend(score_list)
            fold = dict(shared_fold)
            if name == "isolation_forest":
                fold["normal_training_windows"] = len(normal_rows)
                fold["normal_training_aircraft_days"] = len({
                    (row["date_utc"], row["icao24"]) for row in normal_rows
                })
                fold["normal_training_cohort"] = "control_sample"
            fold["validation_roc_auc"] = (
                float(roc_auc_score(y_val, score_list)) if len(set(y_val)) == 2 else None
            )
            fold["validation_average_precision"] = (
                float(average_precision_score(y_val, score_list)) if sum(y_val) else None
            )
            oof[name]["folds"].append(fold)
        fold_counts.append({
            "validation_date": validation_date,
            "purged_aircraft_ids": len(purged),
            "isolation_forest_normal_training_windows": len(normal_rows),
        })

    model_reports: dict[str, Any] = {}
    prediction_paths: dict[str, str] = {}
    for name, values in oof.items():
        model_rows = values["rows"]
        scores = values["scores"]
        labels = target_labels(model_rows, "any_declaration").astype(int).tolist()
        curve, per_date = summarize_curves(model_rows, scores, "any_declaration")
        model_reports[name] = {
            "score_type": values["score_type"],
            "configuration": ISOLATION_FOREST_CONFIG if name == "isolation_forest" else TREE_CONFIG,
            "folds": values["folds"],
            "oof_windows": len(model_rows),
            "oof_events": len(episode_catalog(model_rows)),
            "oof_positive_fraction": float(np.mean(labels)),
            "oof_roc_auc_secondary": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "oof_average_precision_secondary": float(average_precision_score(labels, scores)) if sum(labels) else None,
            "event_signal_subtype_counts": event_subtype_counts(model_rows),
            "recall_vs_false_alert_curve": curve,
            "per_date_results": per_date,
            "uncertainty_note": (
                "Event recall uses nominal Wilson 95% intervals and false-alert rates use nominal Poisson 95% intervals. "
                "These intervals treat events/alerts as independent and may be optimistic under aircraft/day clustering."
            ),
        }
        path = output_dir / f"{name}_oof_predictions.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row, score in zip(model_rows, scores):
                stream.write(json.dumps({
                    "example_id": row["example_id"], "date_utc": row["date_utc"],
                    "icao24": row["icao24"], "group_id": row["group_id"],
                    "anchor_unix_s": row["anchor_unix_s"],
                    "label": row["targets"]["any_declaration"], "targets": row["targets"],
                    "future_events": row["future_events"], "source_cohort": row["source_cohort"],
                    "airborne_exposure_seconds": row["airborne_exposure_seconds"],
                    "score_type": values["score_type"], "score": score,
                }, separators=(",", ":")) + "\n")
        prediction_paths[name] = str(path.resolve())

    bundle_training_rows = [row for row in rows if row.get("source_cohort") == "control_sample"]
    research_model, final_normal_rows = fit_isolation_forest(bundle_training_rows, SEED)
    bundle_path = output_dir / "isolation_forest_research_bundle.joblib"
    bundle = {
        "schema_version": 1,
        "model_type": "isolation_forest",
        "target": EPISODE_TARGET,
        "score_type": "isolation_anomaly_score",
        "score_direction": "higher means more unusual relative to sampled control-flight windows",
        "feature_field": "temporal_features",
        "feature_names": list(TEMPORAL_FEATURE_NAMES),
        "model": research_model,
        "training": {
            "dates": dates,
            "configuration": ISOLATION_FOREST_CONFIG,
            "training_cohort": "control_sample",
            "training_windows": len(final_normal_rows),
            "training_aircraft_days": len({
                (row["date_utc"], row["icao24"]) for row in final_normal_rows
            }),
            "python_version": __import__("platform").python_version(),
            "numpy_version": str(np.__version__),
            "scikit_learn_version": sklearn.__version__,
            "scipy_version": scipy.__version__,
        },
        "warning": (
            "Research-only model trained on all ten development dates. Its score is not an emergency probability; "
            "no untouched final test date or operational threshold is available."
        ),
    }
    joblib.dump(bundle, bundle_path)

    comparison_cap = 1.0
    forest_at_cap = best_recall_at_false_alert_limit(
        model_reports["isolation_forest"]["recall_vs_false_alert_curve"], comparison_cap,
    )
    tree_at_cap = best_recall_at_false_alert_limit(
        model_reports["temporal_boosted_tree_baseline"]["recall_vs_false_alert_curve"], comparison_cap,
    )
    recall_delta = (
        float(forest_at_cap["event_recall"]) - float(tree_at_cap["event_recall"])
        if forest_at_cap["event_recall"] is not None and tree_at_cap["event_recall"] is not None
        else None
    )
    report = {
        "schema_version": 2,
        "target": EPISODE_TARGET,
        "primary_model": "isolation_forest",
        "evaluation": "ten-date leave-one-date-out with aircraft-ID purging; all ten dates are development data",
        "development_dates": dates,
        "alert_curve_grid": {
            "kind": "uniform per-model observed score range plus no-alert point",
            "base_points": CURVE_POINTS,
            "extra_threshold": "immediately above the maximum model score (no-alert point)",
        },
        "no_untouched_final_test_date": True,
        "fold_separation": fold_counts,
        "candidate_audit_and_coverage": audit_context(dataset),
        "isolation_forest": model_reports["isolation_forest"],
        "temporal_boosted_tree_baseline": model_reports["temporal_boosted_tree_baseline"],
        "development_tradeoff_at_1_false_alert_per_1000_control_hours": {
            "descriptive_only": True,
            "isolation_forest": forest_at_cap,
            "temporal_boosted_tree_baseline": tree_at_cap,
            "event_recall_difference": recall_delta,
        },
        "tradeoff_assessment": describe_tradeoff_at_limit(forest_at_cap, tree_at_cap),
        "model_bundle": str(bundle_path.resolve()),
        "oof_prediction_files": prediction_paths,
        "selection": (
            "Isolation Forest is the primary research predictor. Development curves and the fixed-budget "
            "summary are descriptive; no operational model threshold is selected."
        ),
        "score_warning": (
            "Isolation Forest outputs an anomaly score, not a calibrated probability of emergency declaration. "
            "Its predictive value is determined only by the held-out declaration evaluation."
        ),
        "signal_policy": (
            "ADS-B declaration proxies grouped into 60-second multi-label episodes. The Isolation Forest is fit "
            "only on sampled control-flight windows; signal values and labels are not model inputs."
        ),
        "warning": "Exploratory development comparison only. A future alerting claim requires new dates collected after model and threshold are frozen.",
    }
    (output_dir / "temporal_model_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    svg_path = output_dir / "event_recall_vs_false_alerts.svg"
    write_curve_svg(report, svg_path)
    report["pooled_curve_svg"] = str(svg_path.resolve())
    (output_dir / "temporal_model_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "declaration_episode_windows.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "temporal_episode_comparison")
    parser.add_argument("--development-dates", nargs="+", default=None)
    parser.add_argument(
        "--reuse-oof-predictions", action="store_true",
        help="rebuild the report and artifact from complete saved OOF prediction files",
    )
    args = parser.parse_args()
    try:
        result = compare(
            args.dataset,
            args.output_dir,
            tuple(args.development_dates) if args.development_dates else None,
            reuse_oof_predictions=args.reuse_oof_predictions,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps({
        "development_dates": result["development_dates"],
        "model_bundle": result["model_bundle"],
        "report": str((args.output_dir / "temporal_model_comparison.json").resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
