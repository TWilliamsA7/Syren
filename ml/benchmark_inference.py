"""Measure selected research-model CPU inference latency on the target host."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np

from ml.label_windows import DEFAULT_OUTPUT


def benchmark(model_path: Path, dataset: Path, runs: int = 1000, batch_size: int = 1) -> dict:
    if runs <= 0 or batch_size <= 0:
        raise ValueError("Runs and batch size must be positive")
    bundle = joblib.load(model_path)
    rows = []
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= batch_size:
                break
    names = bundle["feature_names"]
    feature_set = bundle["configuration"]["features"]
    field = "features" if feature_set == "base" else "temporal_features"
    sample = np.asarray([[row[field][name] for name in names] for row in rows[:batch_size]], dtype=float)
    if len(sample) != batch_size:
        raise ValueError("Dataset has fewer rows than batch size")
    if not np.isfinite(sample).all():
        raise ValueError("Benchmark sample contains nonfinite features")
    if bundle["scaler"] is not None:
        sample = bundle["scaler"].transform(sample)
    predictor = bundle["model"].predict_proba
    for _ in range(min(100, runs)):
        predictor(sample)
    timings = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        predictor(sample)
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "schema_version": 1, "host": platform.node(), "platform": platform.platform(),
        "machine": platform.machine(), "python": platform.python_version(),
        "model_path": str(model_path.resolve()), "batch_size": batch_size, "runs": runs,
        "latency_ms_p50": float(np.percentile(timings, 50)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "latency_ms_p99": float(np.percentile(timings, 99)),
        "warning": "CPU inference timing only; excludes live ADS-B acquisition and feature extraction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model" / "selected_research_model.joblib")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "declaration_windows.jsonl")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.model, args.dataset, args.runs, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
