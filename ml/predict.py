"""Pure-Python inference for the exported Syren logistic baseline."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    features = json.loads(args.features_json.read_text(encoding="utf-8"))
    print(json.dumps({"target": model["target"], "score": probability(model, features)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
