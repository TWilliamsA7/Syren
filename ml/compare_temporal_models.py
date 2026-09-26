"""Compare one compact causal CNN with the fixed temporal HGB baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from ml.compare_models import fit_predict as fit_tree
from ml.features import TEMPORAL_FEATURE_NAMES
from ml.label_windows import DEFAULT_OUTPUT, EPISODE_TARGET, SUPPORTED_SUBTYPE_HEADS
from ml.sequence_features import (
    CHANNEL_NAMES, NORMALIZED_VALUE_CHANNELS, STEPS, VALUE_MASK_CHANNEL,
)

SEED = 17
CNN_CONFIG = {
    "architecture": f"Conv1d({len(CHANNEL_NAMES)},16,kernel=3)-ReLU-Conv1d(16,32,kernel=3)-ReLU-global-average-pool-dropout-Linear({1 + len(SUPPORTED_SUBTYPE_HEADS)})",
    "epochs": 20,
    "batch_size": 2048,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.15,
    "seed": SEED,
    "fold_seed_policy": "base seed plus zero-based fold index; final all-date fit uses base seed",
    "loss_weighting": "aircraft-day-balanced, class-balanced",
    "output_heads": ["any_declaration", *SUPPORTED_SUBTYPE_HEADS],
}
TREE_CONFIG = {
    "features": "temporal", "weighting": "group_balanced",
    "model": "hist_gradient_boosting", "max_leaf_nodes": 3,
    "max_iter": 100,
}
CURVE_POINTS = 201


class CompactSequenceCNN(nn.Module):
    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(len(CHANNEL_NAMES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, 1 + len(SUPPORTED_SUBTYPE_HEADS)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.features(values)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target") != EPISODE_TARGET or row.get("sequence_schema_version") != 2:
                raise ValueError(f"Target or sequence schema mismatch at {path}:{line_no}")
            sequence = np.asarray(row.get("sequence"), dtype=float)
            if sequence.shape != (STEPS, len(CHANNEL_NAMES)) or not np.isfinite(sequence).all():
                raise ValueError(f"Invalid sequence at {path}:{line_no}: {sequence.shape}")
            if not isinstance(row.get("targets"), dict) or not isinstance(row.get("future_events"), list):
                raise ValueError(f"Missing multi-label targets or episode metadata at {path}:{line_no}")
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


def sequence_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["sequence"] for row in rows], dtype=np.float32)


def fit_normalizer(train_x: np.ndarray) -> dict[str, list[float]]:
    """Fit value-channel statistics using observed training-fold cells only."""
    means: list[float] = []
    scales: list[float] = []
    for channel in NORMALIZED_VALUE_CHANNELS:
        mask_channel = VALUE_MASK_CHANNEL[channel]
        observed = train_x[:, :, mask_channel] > 0.5
        values = train_x[:, :, channel][observed]
        if not len(values):
            mean, scale = 0.0, 1.0
        else:
            mean = float(values.mean())
            scale = float(values.std())
            if not math.isfinite(scale) or scale < 1e-6:
                scale = 1.0
        means.append(mean)
        scales.append(scale)
    return {"channels": list(NORMALIZED_VALUE_CHANNELS), "mean": means, "scale": scales}


def normalize_sequences(values: np.ndarray, normalizer: dict[str, list[float]]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for position, channel in enumerate(normalizer["channels"]):
        mask_channel = VALUE_MASK_CHANNEL[channel]
        valid = result[:, :, mask_channel] > 0.5
        transformed = (result[:, :, channel] - normalizer["mean"][position]) / normalizer["scale"][position]
        result[:, :, channel] = np.where(valid, transformed, 0.0)
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_labels(rows: list[dict[str, Any]], head: str) -> np.ndarray:
    return np.asarray([int(row["targets"][head]) for row in rows], dtype=np.float32)


def balanced_target_weights(rows: list[dict[str, Any]], labels: np.ndarray) -> np.ndarray | None:
    """Balance labels and aircraft-days for one output, if both classes occur."""
    group_counts: Counter[tuple[str, str, int]] = Counter(
        (row["date_utc"], row["icao24"], int(label)) for row, label in zip(rows, labels)
    )
    class_groups = Counter(label for _, _, label in group_counts)
    if set(class_groups) != {0, 1}:
        return None
    weights = np.asarray([
        1.0 / (group_counts[(row["date_utc"], row["icao24"], int(label))] * class_groups[int(label)])
        for row, label in zip(rows, labels)
    ], dtype=np.float32)
    weights *= len(rows) / weights.sum()
    return weights


def train_cnn(
    rows: list[dict[str, Any]], seed: int = SEED,
) -> tuple[CompactSequenceCNN, dict[str, list[float]], list[str]]:
    if {int(row["targets"]["any_declaration"]) for row in rows} != {0, 1}:
        raise ValueError("CNN training split requires both any-declaration labels")
    seed_everything(seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    raw = sequence_matrix(rows)
    normalizer = fit_normalizer(raw)
    x = torch.from_numpy(normalize_sequences(raw, normalizer).transpose(0, 2, 1).copy())
    heads = ["any_declaration", *SUPPORTED_SUBTYPE_HEADS]
    y_np = np.column_stack([target_labels(rows, head) for head in heads])
    y = torch.from_numpy(y_np)
    per_head_weights = [balanced_target_weights(rows, y_np[:, index]) for index in range(len(heads))]
    supported_heads = [head for head, weights in zip(heads, per_head_weights) if weights is not None]
    weight_tensor = torch.from_numpy(np.column_stack([
        weights if weights is not None else np.zeros(len(rows), dtype=np.float32)
        for weights in per_head_weights
    ]))
    active = torch.tensor([weights is not None for weights in per_head_weights], dtype=torch.bool)
    model = CompactSequenceCNN(CNN_CONFIG["dropout"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CNN_CONFIG["learning_rate"],
        weight_decay=CNN_CONFIG["weight_decay"],
    )
    batch_size = CNN_CONFIG["batch_size"]
    model.train()
    for _ in range(CNN_CONFIG["epochs"]):
        order = torch.randperm(len(rows))
        for start in range(0, len(rows), batch_size):
            indices = order[start:start + batch_size]
            logits = model(x[indices])
            losses = nn.functional.binary_cross_entropy_with_logits(logits, y[indices], reduction="none")
            batch_weights = weight_tensor[indices]
            per_head = (losses * batch_weights).sum(dim=0) / batch_weights.sum(dim=0).clamp_min(1e-8)
            loss = per_head[active].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), normalizer, supported_heads


@torch.no_grad()
def predict_cnn(
    model: CompactSequenceCNN, rows: list[dict[str, Any]],
    normalizer: dict[str, list[float]], batch_size: int = 2048,
) -> np.ndarray:
    raw = sequence_matrix(rows)
    values = torch.from_numpy(normalize_sequences(raw, normalizer).transpose(0, 2, 1).copy())
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(values[start:start + batch_size])).cpu().numpy()
        for start in range(0, len(rows), batch_size)
    ], axis=0)


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
    # A dense fixed score-domain grid makes every part of the empirical tradeoff
    # visible without choosing an operating point from these development scores.
    return sorted(set(np.linspace(0.0, 1.0, CURVE_POINTS).tolist()
                      + [math.nextafter(max(scores), math.inf)]))


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
        ("temporal_boosted_tree_baseline", "Temporal boosted tree", "#2563eb"),
        ("compact_sequence_cnn", "Compact sequence CNN", "#c026d3"),
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
        '<text x="100" y="34" font-family="Segoe UI,Arial,sans-serif" font-size="23" font-weight="600" fill="#111827">Event recall vs false alerts</text>',
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


def compare(dataset: Path, output_dir: Path, development_dates: tuple[str, ...] | None = None) -> dict[str, Any]:
    rows = load_rows(dataset)
    observed_dates = sorted({row["date_utc"] for row in rows})
    dates = sorted(set(development_dates)) if development_dates else observed_dates
    if set(dates) != set(observed_dates) or len(dates) != 10:
        raise ValueError(f"Expected all ten development dates; observed={observed_dates}, requested={dates}")
    output_dir.mkdir(parents=True, exist_ok=True)
    oof: dict[str, dict[str, Any]] = {
        "temporal_hgb": {"rows": [], "scores": [], "folds": []},
        "sequence_cnn": {
            "rows": [], "scores": [], "folds": [],
            "head_scores": {head: [] for head in ["any_declaration", *SUPPORTED_SUBTYPE_HEADS]},
        },
    }
    fold_counts = []
    for fold_number, validation_date in enumerate(dates, 1):
        train, validation, purged = purged_date_split(rows, validation_date)
        if {row["label"] for row in train} != {0, 1}:
            raise ValueError(f"Training fold {validation_date} lacks both labels")
        print(f"Fold {fold_number}/10 {validation_date}: train={len(train)} validation={len(validation)} purged_aircraft={len(purged)}", flush=True)
        _, tree_scores = fit_tree(train, validation, TREE_CONFIG)
        cnn, normalizer, supported_heads = train_cnn(train, SEED + fold_number - 1)
        cnn_matrix = predict_cnn(cnn, validation, normalizer)
        y_val = target_labels(validation, "any_declaration").astype(int).tolist()
        any_scores = [float(value) for value in cnn_matrix[:, 0]]
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
            "training_seed": SEED + fold_number - 1,
        }
        tree_score_list = [float(value) for value in tree_scores]
        oof["temporal_hgb"]["rows"].extend(validation)
        oof["temporal_hgb"]["scores"].extend(tree_score_list)
        oof["temporal_hgb"]["folds"].append({
            **shared_fold,
            "validation_roc_auc": float(roc_auc_score(y_val, tree_score_list)) if len(set(y_val)) == 2 else None,
            "validation_average_precision": float(average_precision_score(y_val, tree_score_list)) if sum(y_val) else None,
        })

        cnn_fold_heads = {}
        all_heads = ["any_declaration", *SUPPORTED_SUBTYPE_HEADS]
        for column, head in enumerate(all_heads):
            head_scores = [float(value) for value in cnn_matrix[:, column]]
            head_y = target_labels(validation, head).astype(int).tolist()
            head_events = len(episode_catalog(validation, head))
            cnn_fold_heads[head] = {
                "validation_positive_windows": sum(head_y),
                "validation_events": head_events,
                "trained_in_fold": head in supported_heads,
                "validation_roc_auc": float(roc_auc_score(head_y, head_scores)) if len(set(head_y)) == 2 else None,
                "validation_average_precision": float(average_precision_score(head_y, head_scores)) if sum(head_y) else None,
            }
            oof["sequence_cnn"]["head_scores"][head].extend(head_scores)
        oof["sequence_cnn"]["rows"].extend(validation)
        oof["sequence_cnn"]["scores"].extend(any_scores)
        oof["sequence_cnn"]["folds"].append({
            **shared_fold,
            "validation_roc_auc": cnn_fold_heads["any_declaration"]["validation_roc_auc"],
            "validation_average_precision": cnn_fold_heads["any_declaration"]["validation_average_precision"],
            "training_normalizer": normalizer,
            "supported_training_heads": supported_heads,
            "head_scores": cnn_fold_heads,
        })
        fold_counts.append({"validation_date": validation_date, "purged_aircraft_ids": len(purged)})

    model_reports: dict[str, Any] = {}
    prediction_paths: dict[str, str] = {}
    for name, values in oof.items():
        model_rows = values["rows"]
        scores = values["scores"]
        labels = target_labels(model_rows, "any_declaration").astype(int).tolist()
        curve, per_date = summarize_curves(model_rows, scores, "any_declaration")
        y_mean = float(np.mean(labels))
        model_reports[name] = {
            "configuration": TREE_CONFIG if name == "temporal_hgb" else CNN_CONFIG,
            "folds": values["folds"],
            "oof_windows": len(model_rows),
            "oof_events": len(episode_catalog(model_rows)),
            "oof_positive_fraction": y_mean,
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
            for index, (row, score) in enumerate(zip(model_rows, scores)):
                prediction = {
                    "example_id": row["example_id"], "date_utc": row["date_utc"],
                    "icao24": row["icao24"], "group_id": row["group_id"],
                    "anchor_unix_s": row["anchor_unix_s"],
                    "label": row["targets"]["any_declaration"], "targets": row["targets"],
                    "future_events": row["future_events"], "source_cohort": row["source_cohort"],
                    "airborne_exposure_seconds": row["airborne_exposure_seconds"],
                    "score": score,
                }
                if name == "sequence_cnn":
                    prediction["head_scores"] = {
                        head: values[index] for head, values in oof[name]["head_scores"].items()
                    }
                stream.write(json.dumps({
                    **prediction,
                }, separators=(",", ":")) + "\n")
        prediction_paths[name] = str(path.resolve())

    subtype_reports = {}
    for head, scores in oof["sequence_cnn"]["head_scores"].items():
        head_curve, head_per_date = summarize_curves(oof["sequence_cnn"]["rows"], scores, head)
        head_labels = target_labels(oof["sequence_cnn"]["rows"], head).astype(int).tolist()
        subtype_reports[head] = {
            "oof_positive_windows": sum(head_labels),
            "oof_events": len(episode_catalog(oof["sequence_cnn"]["rows"], head)),
            "oof_roc_auc_secondary": float(roc_auc_score(head_labels, scores)) if len(set(head_labels)) == 2 else None,
            "oof_average_precision_secondary": float(average_precision_score(head_labels, scores)) if sum(head_labels) else None,
            "recall_vs_false_alert_curve": head_curve,
            "per_date_results": head_per_date,
        }
    model_reports["sequence_cnn"]["subtype_heads"] = subtype_reports

    # Refit the fixed CNN on all development dates for an isolated research bundle.
    research_model, research_normalizer, research_supported_heads = train_cnn(rows, SEED)
    bundle_path = output_dir / "sequence_cnn_research_bundle.pt"
    torch.save({
        "schema_version": 2,
        "target": EPISODE_TARGET,
        "output_heads": ["any_declaration", *SUPPORTED_SUBTYPE_HEADS],
        "architecture": CNN_CONFIG["architecture"],
        "model_state_dict": research_model.state_dict(),
        "preprocessing": {
            "channel_names": list(CHANNEL_NAMES),
            "normalizer": research_normalizer,
            "sequence_steps": STEPS,
            "interval_seconds": 10,
            "history_seconds": 300,
            "max_interpolation_gap_seconds": 30,
            "track_representation": "sin_cos of shortest-arc interpolation",
            "missing_representation": "zero-filled after fold training normalization plus explicit masks",
            "position_representation": "causal anchor-relative local east/north offsets in nautical miles; absolute coordinates are not inputs",
        },
        "training": {
            "dates": dates, "configuration": CNN_CONFIG,
            "supported_heads": research_supported_heads,
            "training_aircraft_days": len({(row["date_utc"], row["icao24"]) for row in rows}),
            "training_windows": len(rows),
            "python_version": __import__("platform").python_version(),
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "scikit_learn_version": sklearn.__version__, "scipy_version": scipy.__version__,
        },
        "warning": "Research-only bundle trained and compared on all ten development dates. Not connected to deployed inference. No untouched test date.",
    }, bundle_path)

    report = {
        "schema_version": 1,
        "target": EPISODE_TARGET,
        "evaluation": "ten-date leave-one-date-out with aircraft-ID purging; all ten dates are development data",
        "development_dates": dates,
        "alert_curve_grid": {
            "kind": "uniform score-domain grid",
            "base_points": CURVE_POINTS,
            "step": 1.0 / (CURVE_POINTS - 1),
            "extra_threshold": "immediately above the maximum model score (no-alert point)",
        },
        "no_untouched_final_test_date": True,
        "fold_separation": fold_counts,
        "candidate_audit_and_coverage": audit_context(dataset),
        "temporal_boosted_tree_baseline": model_reports["temporal_hgb"],
        "compact_sequence_cnn": model_reports["sequence_cnn"],
        "model_bundle": str(bundle_path.resolve()),
        "oof_prediction_files": prediction_paths,
        "selection": "No operational model or threshold selected from these development scores. Compare event-level tradeoffs across dates; AUC is secondary.",
        "signal_policy": "ADS-B declaration proxies grouped into 60-second multi-label episodes. The CNN predicts any declaration plus four common subtype heads; rarer subtype labels remain audit/evaluation metadata. No signal or subtype value is an input feature.",
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
    args = parser.parse_args()
    try:
        result = compare(args.dataset, args.output_dir, tuple(args.development_dates) if args.development_dates else None)
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
