"""One-date pilot of a causal, stateful warning score on ADS-B behavior.

This is a research replay, not an operational alert threshold. It deliberately
excludes transponder/emergency status because the target is advance warning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ml.event_supervised import (
    FALSE_ALERT_BUDGET_PER_1000H,
    PILOT_DATE,
    cutoff_from_training_controls,
    load_compact_dataset,
    pilot_split,
    replay_validation,
)
from ml.features import TEMPORAL_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/learning/ten_day_development/declaration_episode_windows.jsonl"
OUTPUT = ROOT / "data/learning/ten_day_development/temporal_risk_pilot"
FEATURE_INDEX = {name: index for index, name in enumerate(TEMPORAL_FEATURE_NAMES)}
EVIDENCE_HALF_LIFE_SECONDS = 180.0
MAX_GROUP_GAP_SECONDS = 180.0


def behavior_evidence(vector: np.ndarray) -> dict[str, float]:
    """Independent, interpretable signals from the preceding five minutes."""
    f = lambda name: float(vector[FEATURE_INDEX[name]])
    if f("airborne_at_anchor") < 0.5:
        return {}
    evidence: dict[str, float] = {}
    if f("vertical_rate_valid_fraction") >= 0.5:
        descent = -f("vertical_rate_last_fpm")
        if descent >= 3000:
            evidence["rapid_descent"] = min(2.5, 1.0 + (descent - 3000) / 2000)
        recent = f("vertical_rate_mean_fpm_30s")
        prior = f("vertical_rate_mean_fpm_180s")
        if recent < -700 and prior - recent >= 900:
            evidence["worsening_descent"] = min(1.8, (prior - recent) / 900)
    if f("speed_valid_fraction") >= 0.5 and f("speed_last_kt") >= 70:
        decay = -f("speed_trend_kt_s_60s")
        if decay >= 0.35:
            evidence["speed_loss"] = min(2.0, decay / 0.35)
    if f("track_valid_fraction") >= 0.5:
        turn = f("turn_rate_deg_s_60s")
        if turn >= 0.5:
            evidence["sustained_turn"] = min(2.0, turn / 0.5)
    if f("altitude_valid_fraction") >= 0.5:
        if f("altitude_range_ft") >= 1800 and abs(f("altitude_delta_ft")) < 0.5 * f("altitude_range_ft"):
            evidence["altitude_reversal"] = 1.0
    if f("max_observation_gap_s") >= 75:
        evidence["telemetry_gap"] = 0.5
    return evidence


def score_stream(matrix: np.ndarray, rows: list[dict[str, Any]], indices: list[int]) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Decay evidence by elapsed time, keeping aircraft/flight groups separate."""
    if not np.isfinite(matrix[indices]).all():
        raise ValueError("Nonfinite input features")
    ordered = sorted(enumerate(indices), key=lambda pair: (rows[pair[1]]["group_id"], rows[pair[1]]["anchor_unix_s"]))
    result = np.zeros(len(indices), dtype=float)
    explanations: list[dict[str, float]] = [{} for _ in indices]
    state: dict[str, tuple[float, dict[str, float]]] = {}
    for position, index in ordered:
        row = rows[index]
        group = row["group_id"]
        now = row["anchor_unix_s"]
        previous = state.get(group)
        if previous is None or now - previous[0] > MAX_GROUP_GAP_SECONDS:
            active: dict[str, float] = {}
        else:
            if now < previous[0]:
                raise ValueError("Out-of-order group observations")
            factor = 0.5 ** ((now - previous[0]) / EVIDENCE_HALF_LIFE_SECONDS)
            active = {name: strength * factor for name, strength in previous[1].items()
                      if strength * factor >= 0.05}
        current = behavior_evidence(matrix[index])
        for name, strength in current.items():
            # Repeated behavior builds evidence, with a per-signal cap.
            active[name] = min(3.0, active.get(name, 0.0) + strength)
        state[group] = (now, active)
        strong = sorted(active.values(), reverse=True)
        # Two independent signal types are required for a predictive score.
        result[position] = sum(strong[:3]) if len(strong) >= 2 else 0.0
        explanations[position] = dict(active)
    return result, explanations


def run_pilot(dataset: Path = DATASET, output_dir: Path = OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, rows = load_compact_dataset(dataset, dataset.parent / "event_supervised/compact_episode_cache.joblib")
    training, validation = pilot_split(rows)
    train_controls = [index for index in training if rows[index]["source_cohort"] == "control_sample"]
    training_scores, _ = score_stream(matrix, rows, training)
    score_by_index = dict(zip(training, training_scores))
    control_scores = np.asarray([score_by_index[index] for index in train_controls])
    cutoff = cutoff_from_training_controls(rows, train_controls, control_scores)
    validation_scores, explanations = score_stream(matrix, rows, validation)
    replay = replay_validation(rows, validation, validation_scores, float(cutoff["score_cutoff"]))
    alert_details = [{"icao24": rows[index]["icao24"], "anchor_unix_s": rows[index]["anchor_unix_s"],
                      "score": float(score), "signals": signals}
                     for index, score, signals in zip(validation, validation_scores, explanations)
                     if score >= cutoff["score_cutoff"]]
    limit = math.floor(FALSE_ALERT_BUDGET_PER_1000H * replay["control_hours"] / 1000)
    passed = (replay["detected_events"] >= 3 and replay["detected_aircraft"] >= 2
              and replay["control_false_alerts"] <= limit)
    report = {
        "pilot_only": True,
        "validation_date": PILOT_DATE,
        "score": "decayed, multi-signal behavior evidence; not an emergency probability",
        "cutoff_from_training_controls": cutoff,
        "validation_false_alert_limit": limit,
        "passed_pilot_gate": passed,
        "validation": replay,
        "above_cutoff_window_examples": alert_details[:25],
        "note": "One development date only. No operational threshold or future-date claim.",
    }
    path = output_dir / f"pilot_{PILOT_DATE}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    outcome = run_pilot(arguments.dataset, arguments.output_dir)
    validation = outcome["validation"]
    print(json.dumps({"passed": outcome["passed_pilot_gate"],
                      "events": validation["detected_events"],
                      "event_count": validation["event_count"],
                      "false_alerts": validation["control_false_alerts"],
                      "false_alert_limit": outcome["validation_false_alert_limit"]}))
