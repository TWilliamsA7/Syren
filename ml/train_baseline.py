#!/usr/bin/env python3
"""Train a date-held-out, deployable logistic baseline on labeled windows."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ml.features import FEATURE_NAMES
from ml.label_windows import DEFAULT_OUTPUT, TARGET
from ml.predict import probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "declaration_windows.jsonl")
    parser.add_argument("--summary", type=Path, default=DEFAULT_OUTPUT / "declaration_windows_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "model")
    parser.add_argument("--holdout-date", help="UTC date to hold out; defaults to latest available date")
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target") != TARGET or row.get("label") not in {0, 1}:
                raise ValueError(f"Invalid target or label at {path}:{line_number}")
            features = row.get("features")
            if not isinstance(features, dict) or tuple(features) != FEATURE_NAMES:
                raise ValueError(f"Feature schema mismatch at {path}:{line_number}")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in features.values()
            ):
                raise ValueError(f"Nonfinite feature at {path}:{line_number}")
            rows.append(row)
    return rows


def group_weights(rows: list[dict[str, Any]]) -> list[float]:
    """Give each aircraft-day and label an equal contribution within its class."""
    counts = Counter((row["date_utc"], row["icao24"], row["label"]) for row in rows)
    class_groups = Counter(label for _, _, label in counts)
    if set(class_groups) != {0, 1}:
        raise ValueError("Training split needs positive and negative aircraft-days")
    return [
        1.0 / (counts[(row["date_utc"], row["icao24"], row["label"])] * class_groups[row["label"]])
        for row in rows
    ]


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies from requirements.txt") from exc

    rows = load_dataset(args.dataset)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("target") != TARGET or summary.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Dataset and summary schema disagree")
    dates = sorted({row["date_utc"] for row in rows})
    if len(dates) < 2:
        raise ValueError("At least two completed dates are required for a date-held-out evaluation")
    holdout = args.holdout_date or dates[-1]
    if holdout not in dates:
        raise ValueError(f"Holdout date {holdout} not present in dataset")
    test = [row for row in rows if row["date_utc"] == holdout]
    test_aircraft = {row["icao24"] for row in test}
    train_rows = [row for row in rows if row["date_utc"] != holdout and row["icao24"] not in test_aircraft]
    if not train_rows or not test or {row["label"] for row in train_rows} != {0, 1} or {row["label"] for row in test} != {0, 1}:
        raise ValueError("Both train and holdout must contain positives and negatives")

    def matrix(selected: list[dict[str, Any]]) -> list[list[float]]:
        return [[float(row["features"][name]) for name in FEATURE_NAMES] for row in selected]

    x_train = matrix(train_rows)
    y_train = [row["label"] for row in train_rows]
    x_test = matrix(test)
    y_test = [row["label"] for row in test]
    weights = group_weights(train_rows)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipeline.fit(x_train, y_train, logisticregression__sample_weight=weights)
    scores = pipeline.predict_proba(x_test)[:, 1]
    scaler = pipeline.named_steps["standardscaler"]
    estimator = pipeline.named_steps["logisticregression"]
    model = {
        "schema_version": 1,
        "target": TARGET,
        "feature_names": list(FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "label_config": summary["config"],
        "trained_on_dates": sorted({row["date_utc"] for row in train_rows}),
        "held_out_date": holdout,
        "sklearn_version": sklearn.__version__,
        "score_warning": "Score is a ranking score on an enriched aircraft-day sample, not a calibrated real-world emergency probability.",
    }
    export_difference = max(
        abs(float(score) - probability(model, row["features"]))
        for row, score in zip(test, scores)
    )
    if export_difference > 1e-9:
        raise ValueError(f"Exported JSON inference disagrees with scikit-learn: {export_difference}")
    metrics = {
        "schema_version": 1,
        "target": TARGET,
        "train_dates": model["trained_on_dates"],
        "holdout_date": holdout,
        "purged_train_aircraft_present_in_holdout": len({row["icao24"] for row in rows if row["date_utc"] != holdout} & test_aircraft),
        "train_windows": len(train_rows),
        "train_positive_windows": sum(y_train),
        "train_positive_aircraft_days": len({(row["date_utc"], row["icao24"]) for row in train_rows if row["label"]}),
        "holdout_windows": len(test),
        "holdout_positive_windows": sum(y_test),
        "holdout_positive_aircraft_days": len({(row["date_utc"], row["icao24"]) for row in test if row["label"]}),
        "holdout_roc_auc": float(roc_auc_score(y_test, scores)),
        "holdout_average_precision": float(average_precision_score(y_test, scores)),
        "holdout_window_positive_fraction": sum(y_test) / len(y_test),
        "export_max_abs_score_difference": export_difference,
        "warning": "These window metrics are from three sampled days with correlated windows and enriched positives. They do not establish operational accuracy or calibrated risk.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "holdout_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row, score in zip(test, scores):
            stream.write(json.dumps({
                "example_id": row["example_id"],
                "group_id": row["group_id"],
                "label": row["label"],
                "score": float(score),
            }, separators=(",", ":")) + "\n")
    return metrics


def main() -> int:
    args = parse_args()
    try:
        metrics = train(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Training failed: {exc}")
        return 1
    print(json.dumps(metrics, indent=2))
    print(f"Model and holdout predictions: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
