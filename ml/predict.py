"""Score a legacy logistic export or research Isolation Forest bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def probability(model: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    names = model["feature_names"]
    coefficients = model["coefficients"]
    means = model["mean"]
    scales = model["scale"]
    if not (len(names) == len(coefficients) == len(means) == len(scales)):
        raise ValueError("Model feature schema is inconsistent")
    if set(features) != set(names):
        missing = set(names) - set(features)
        extra = set(features) - set(names)
        raise ValueError(f"Feature mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    score = float(model["intercept"])
    for name, weight, mean, scale in zip(names, coefficients, means, scales):
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise ValueError(f"Feature {name} must be finite numeric")
        if scale <= 0:
            raise ValueError(f"Invalid scale for {name}")
        score += float(weight) * (float(value) - float(mean)) / float(scale)
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exponent = math.exp(score)
    return exponent / (1.0 + exponent)


def isolation_anomaly_score(bundle: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    """Score one temporal feature mapping; larger values indicate more unusual behavior.

    This is an Isolation Forest anomaly score, not an emergency probability.
    """
    if bundle.get("model_type") != "isolation_forest":
        raise ValueError("Model bundle is not an Isolation Forest")
    names = bundle.get("feature_names")
    model = bundle.get("model")
    if not isinstance(names, list) or not names or model is None:
        raise ValueError("Isolation Forest bundle is missing its feature schema or model")
    if set(features) != set(names):
        missing = set(names) - set(features)
        extra = set(features) - set(names)
        raise ValueError(f"Feature mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    values = []
    for name in names:
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise ValueError(f"Feature {name} must be finite numeric")
        values.append(float(value))
    # sklearn's decision_function is larger for inliers; invert it so this API
    # consistently reports larger values for more anomalous windows.
    return -float(model.decision_function([values])[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    args = parser.parse_args()
    features = json.loads(args.features_json.read_text(encoding="utf-8"))
    if args.model.suffix.lower() == ".joblib":
        import joblib

        bundle = joblib.load(args.model)
        score = isolation_anomaly_score(bundle, features)
        result = {
            "target": bundle["target"],
            "score_type": bundle["score_type"],
            "anomaly_score": score,
            "warning": "Anomaly score only; not a probability of an emergency declaration.",
        }
    else:
        model = json.loads(args.model.read_text(encoding="utf-8"))
        result = {
            "target": model["target"],
            "score": probability(model, features),
            "score_type": "legacy_logistic_score",
        }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
