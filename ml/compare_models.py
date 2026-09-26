"""Aircraft-disjoint leave-one-date-out development comparison on proxy labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ml.evaluate_alerts import conservative_threshold, replay
from ml.features import FEATURE_NAMES, TEMPORAL_FEATURE_NAMES
from ml.label_windows import DEFAULT_OUTPUT
from ml.train_baseline import group_weights, load_dataset

DEFAULT_DEVELOPMENT_DATES = ("2024-02-15", "2024-05-15", "2024-08-15")


def configurations() -> list[dict]:
    result = []
    for feature_set in ("base", "temporal"):
        for weighting in ("uniform", "group_balanced", "group_balanced_pos2", "group_balanced_pos4"):
            for c in (0.1, 1.0, 10.0):
                result.append({"features": feature_set, "weighting": weighting, "model": "logistic", "C": c})
            for leaves in (3, 7):
                for iterations in (50, 100):
                    result.append({"features": feature_set, "weighting": weighting,
                                   "model": "hist_gradient_boosting", "max_leaf_nodes": leaves,
                                   "max_iter": iterations})
    return result


def weights_for(rows: list[dict], mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(rows))
    weights = np.asarray(group_weights(rows), dtype=float)
    if mode == "group_balanced_pos2":
        weights *= np.where(np.asarray([row["label"] for row in rows]) == 1, 2, 1)
    elif mode == "group_balanced_pos4":
        weights *= np.where(np.asarray([row["label"] for row in rows]) == 1, 4, 1)
    elif mode != "group_balanced":
        raise ValueError(f"Unknown weighting: {mode}")
    return weights * len(rows) / weights.sum()


def array(rows: list[dict], feature_set: str) -> np.ndarray:
    names = FEATURE_NAMES if feature_set == "base" else TEMPORAL_FEATURE_NAMES
    field = "features" if feature_set == "base" else "temporal_features"
    return np.asarray([[row[field][name] for name in names] for row in rows], dtype=float)


def fit_predict(train: list[dict], validation: list[dict], config: dict) -> tuple[np.ndarray, np.ndarray]:
    x = array(train, config["features"])
    xv = array(validation, config["features"])
    y = np.asarray([row["label"] for row in train])
    weights = weights_for(train, config["weighting"])
    if config["model"] == "logistic":
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        xv = scaler.transform(xv)
        model = LogisticRegression(C=config["C"], max_iter=2000, solver="lbfgs")
    else:
        model = HistGradientBoostingClassifier(
            max_leaf_nodes=config["max_leaf_nodes"], max_iter=config["max_iter"],
            learning_rate=0.05, min_samples_leaf=100, l2_regularization=1,
            early_stopping=False, random_state=17,
        )
    model.fit(x, y, sample_weight=weights)
    return model.predict_proba(x)[:, 1], model.predict_proba(xv)[:, 1]


def wilson(successes: int, trials: int) -> list[float] | None:
    if not trials:
        return None
    z = 1.959963984540054
    p = successes / trials
    d = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / d
    span = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return [center - span, center + span]


def poisson_rate_interval(count: int, hours: float) -> list[float] | None:
    if hours <= 0:
        return None
    lower = 0.5 * chi2.ppf(0.025, 2 * count) if count else 0.0
    upper = 0.5 * chi2.ppf(0.975, 2 * (count + 1))
    return [lower * 1000 / hours, upper * 1000 / hours]


def select_development_rows(rows: list[dict], development_dates: tuple[str, ...]) -> list[dict]:
    selected = [row for row in rows if row["date_utc"] in development_dates]
    dates = {row["date_utc"] for row in selected}
    if dates != set(development_dates) or len(dates) < 3:
        raise ValueError(f"Expected at least three explicitly named development dates, got {sorted(dates)}")
    return selected


def compare(dataset: Path, output_dir: Path, development_dates: tuple[str, ...] = DEFAULT_DEVELOPMENT_DATES) -> dict:
    all_rows = load_dataset(dataset)
    rows = select_development_rows(all_rows, development_dates)
    dates = sorted({row["date_utc"] for row in rows})
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    best = None
    best_predictions = None
    for number, config in enumerate(configurations(), 1):
        oof_rows: list[dict] = []
        oof_scores: list[float] = []
        folds = []
        for date in dates:
            validation = [row for row in rows if row["date_utc"] == date]
            aircraft = {row["icao24"] for row in validation}
            train = [row for row in rows if row["date_utc"] != date and row["icao24"] not in aircraft]
            if {row["label"] for row in train} != {0, 1} or {row["label"] for row in validation} != {0, 1}:
                raise ValueError(f"Both classes required in date fold {date}")
            train_scores, val_scores = fit_predict(train, validation, config)
            folds.append({
                "validation_date": date, "train_aircraft_days": len({(r["date_utc"], r["icao24"]) for r in train}),
                "train_positive_aircraft_days": len({(r["date_utc"], r["icao24"]) for r in train if r["label"]}),
                "train_auc": float(roc_auc_score([r["label"] for r in train], train_scores)),
                "validation_auc": float(roc_auc_score([r["label"] for r in validation], val_scores)),
                "validation_ap": float(average_precision_score([r["label"] for r in validation], val_scores)),
            })
            oof_rows.extend(validation)
            oof_scores.extend(float(score) for score in val_scores)
        y = [row["label"] for row in oof_rows]
        threshold = conservative_threshold(oof_rows, oof_scores)
        alert = replay(oof_rows, oof_scores, threshold)
        item = {
            "configuration": config, "folds": folds,
            "oof_roc_auc": float(roc_auc_score(y, oof_scores)),
            "oof_average_precision": float(average_precision_score(y, oof_scores)),
            "oof_positive_fraction": sum(y) / len(y),
            "alert_at_conservative_1_per_1000h_threshold": alert,
        }
        comparisons.append(item)
        rank = (alert["event_recall"] or 0, item["oof_average_precision"], item["oof_roc_auc"])
        if best is None or rank > best[0]:
            best = (rank, item)
            best_predictions = (oof_rows, oof_scores)
        print(f"{number}/{len(configurations())} {config} AUC={item['oof_roc_auc']:.3f} AP={item['oof_average_precision']:.3f}", flush=True)
    assert best is not None and best_predictions is not None
    best_rows, best_scores = best_predictions
    best_alert = best[1]["alert_at_conservative_1_per_1000h_threshold"]
    best_alert["event_recall_wilson_95"] = wilson(best_alert["detected_events"], best_alert["event_count"])
    best_alert["false_alert_rate_poisson_95_per_1000h"] = poisson_rate_interval(
        best_alert["control_false_alerts"], best_alert["control_observed_airborne_hours"])
    control_scores = np.asarray([score for row, score in zip(best_rows, best_scores)
                                 if row["source_cohort"] == "control_sample"])
    thresholds = sorted(set(float(np.quantile(control_scores, q)) for q in
                            (0, .5, .8, .9, .95, .98, .99, .995, .999, .9995, 1)))
    thresholds.append(math.nextafter(float(control_scores.max()), math.inf))
    curve = [replay(best_rows, best_scores, threshold) for threshold in thresholds]
    report = {
        "schema_version": 1, "target": "first_observed_adsb_emergency_declaration",
        "evaluation": "exploratory aircraft-disjoint leave-one-date-out; no untouched test date",
        "candidate_count": len(comparisons), "development_dates": dates,
        "selection_rule": "highest event recall at conservative <=1/1000 control flight-hour threshold, then AP, then ROC AUC",
        "best_exploratory": best[1], "best_alert_curve": curve,
        "warning": "Threshold and configuration selected on the same development folds; do not use as an operational claim.",
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "all_configurations.jsonl").open("w", encoding="utf-8") as stream:
        for item in comparisons:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    with (output_dir / "best_oof_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, score in zip(best_rows, best_scores):
            stream.write(json.dumps({
                "example_id": row["example_id"], "date_utc": row["date_utc"],
                "icao24": row["icao24"], "group_id": row["group_id"],
                "anchor_unix_s": row["anchor_unix_s"], "event_unix_s": row["event_unix_s"],
                "label": row["label"], "source_cohort": row["source_cohort"],
                "airborne_exposure_seconds": row["airborne_exposure_seconds"], "score": score,
            }, separators=(",", ":")) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "declaration_windows.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "comparison")
    parser.add_argument("--development-dates", nargs="+", default=list(DEFAULT_DEVELOPMENT_DATES),
                        help="Only these dates participate in model selection; other dates remain untouched")
    args = parser.parse_args()
    result = compare(args.dataset, args.output_dir, tuple(args.development_dates))
    print(json.dumps({"best_exploratory": result["best_exploratory"],
                      "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
