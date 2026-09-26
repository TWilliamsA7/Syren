"""Cluster-bootstrap exploratory alert metrics by whole aircraft-days."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.evaluate_alerts import replay
from ml.label_windows import DEFAULT_OUTPUT


def percentile_interval(values: list[float]) -> list[float] | None:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None


def bootstrap(comparison_dir: Path, iterations: int = 200, seed: int = 17) -> dict:
    report = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    with (comparison_dir / "best_oof_predictions.jsonl").open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream if line.strip()]
    rows = [{key: value for key, value in prediction.items() if key != "score"} for prediction in predictions]
    scores = np.asarray([prediction["score"] for prediction in predictions], dtype=float)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["date_utc"], row["icao24"])].append(index)
    dates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in groups:
        dates[key[0]].append(key)
    threshold = report["best_exploratory"]["alert_at_conservative_1_per_1000h_threshold"]["threshold"]
    contributions = {}
    for key, indices in groups.items():
        one = replay([rows[i] for i in indices], [float(scores[i]) for i in indices], threshold)
        contributions[key] = (one["detected_events"], one["event_count"],
                              one["control_false_alerts"], one["control_observed_airborne_hours"])
    rng = np.random.default_rng(seed)
    distributions = {"roc_auc": [], "average_precision": [], "event_recall": [],
                     "control_false_alerts_per_1000_hours": []}
    for _ in range(iterations):
        sampled = [dates[day][int(index)] for day in sorted(dates)
                   for index in rng.integers(0, len(dates[day]), size=len(dates[day]))]
        multiplicity = defaultdict(int)
        for key in sampled:
            multiplicity[key] += 1
        weights = np.zeros(len(rows))
        for key, count in multiplicity.items():
            weights[groups[key]] = count
        if len(set(labels[weights > 0])) == 2:
            distributions["roc_auc"].append(float(roc_auc_score(labels, scores, sample_weight=weights)))
            distributions["average_precision"].append(float(
                average_precision_score(labels, scores, sample_weight=weights)))
        detected = sum(contributions[key][0] * count for key, count in multiplicity.items())
        events = sum(contributions[key][1] * count for key, count in multiplicity.items())
        alerts = sum(contributions[key][2] * count for key, count in multiplicity.items())
        hours = sum(contributions[key][3] * count for key, count in multiplicity.items())
        if events:
            distributions["event_recall"].append(detected / events)
        if hours:
            distributions["control_false_alerts_per_1000_hours"].append(alerts * 1000 / hours)
    intervals = {name: percentile_interval(values) for name, values in distributions.items()}
    if distributions["control_false_alerts_per_1000_hours"] and max(
        distributions["control_false_alerts_per_1000_hours"]) == 0:
        intervals["control_false_alerts_per_1000_hours"] = None
    output = {
        "schema_version": 1, "iterations": iterations, "seed": seed,
        "resampling_unit": "aircraft-day, stratified by date", "fixed_development_threshold": threshold,
        "cluster_bootstrap_95": intervals,
        "warning": "Exploratory interval conditional on three selected dates and an already selected model/threshold; does not measure new-date uncertainty. Zero observed control alerts cannot yield a useful bootstrap upper bound; use the Poisson interval in comparison.json.",
    }
    (comparison_dir / "cluster_uncertainty.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_OUTPUT / "comparison")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.comparison_dir, args.iterations), indent=2))


if __name__ == "__main__":
    main()
