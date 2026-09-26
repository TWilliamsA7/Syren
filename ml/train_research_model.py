"""Fit the best development configuration on all sampled days for research replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.compare_models import array, weights_for
from ml.features import FEATURE_NAMES, TEMPORAL_FEATURE_NAMES
from ml.label_windows import DEFAULT_OUTPUT, TARGET
from ml.predict import probability
from ml.train_baseline import load_dataset


def train(dataset: Path, comparison_dir: Path, output_dir: Path) -> dict:
    comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    config = comparison["best_exploratory"]["configuration"]
    development_dates = set(comparison["development_dates"])
    rows = [row for row in load_dataset(dataset) if row["date_utc"] in development_dates]
    if {row["date_utc"] for row in rows} != development_dates:
        raise ValueError("Selected development dates are absent from the dataset")
    x = array(rows, config["features"])
    y = np.asarray([row["label"] for row in rows])
    weights = weights_for(rows, config["weighting"])
    scaler = None
    if config["model"] == "logistic":
        scaler = StandardScaler().fit(x)
        x_model = scaler.transform(x)
        model = LogisticRegression(C=config["C"], max_iter=2000, solver="lbfgs")
    else:
        x_model = x
        model = HistGradientBoostingClassifier(
            max_leaf_nodes=config["max_leaf_nodes"], max_iter=config["max_iter"],
            learning_rate=0.05, min_samples_leaf=100, l2_regularization=1,
            early_stopping=False, random_state=17,
        )
    model.fit(x_model, y, sample_weight=weights)
    names = FEATURE_NAMES if config["features"] == "base" else TEMPORAL_FEATURE_NAMES
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_names": list(names),
                 "target": TARGET, "configuration": config}, output_dir / "selected_research_model.joblib")
    portable = None
    if config["model"] == "logistic":
        portable_model = {
            "schema_version": 1, "target": TARGET, "feature_names": list(names),
            "feature_set": config["features"], "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "warning": "Research proxy model, not a calibrated real-world emergency predictor.",
        }
        field = "features" if config["features"] == "base" else "temporal_features"
        mismatch = max(abs(probability(portable_model, row[field]) - float(score))
                       for row, score in zip(rows[:100], model.predict_proba(x_model[:100])[:, 1]))
        if mismatch > 1e-9:
            raise ValueError(f"Portable inference mismatch: {mismatch}")
        portable = output_dir / "selected_research_model.json"
        portable.write_text(json.dumps(portable_model, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": 1, "target": TARGET, "configuration": config,
        "training_dates": sorted({row["date_utc"] for row in rows}),
        "training_aircraft_days": len({(row["date_utc"], row["icao24"]) for row in rows}),
        "positive_aircraft_days": len({(row["date_utc"], row["icao24"]) for row in rows if row["label"]}),
        "python_requirement": "CPython 3.13.3", "sklearn_version": sklearn.__version__,
        "model_file": str((output_dir / "selected_research_model.joblib").resolve()),
        "portable_json_file": str(portable.resolve()) if portable else None,
        "warning": "Best configuration and threshold were chosen on the same three development days. Research replay only; no untouched evaluation or Orin benchmark.",
    }
    (output_dir / "selected_research_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "declaration_windows.jsonl")
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_OUTPUT / "comparison")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "model")
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.comparison_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
