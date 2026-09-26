"""Replay window scores as flight-level alerts; development metrics only."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence


def event_key(row: dict[str, Any]) -> tuple[str, str, float] | None:
    event = row.get("event_unix_s")
    return (row["date_utc"], row["icao24"], float(event)) if event is not None else None


def replay(
    rows: Sequence[dict[str, Any]], scores: Sequence[float], threshold: float,
    suppression_seconds: float = 600,
) -> dict[str, Any]:
    if len(rows) != len(scores):
        raise ValueError("Rows and scores have different lengths")
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    events = {key for row in rows if row["label"] and (key := event_key(row)) is not None}
    exposure = sum(float(row.get("airborne_exposure_seconds", 0.0))
                   for row in rows if row.get("source_cohort") == "control_sample") / 3600
    for row, score in zip(rows, scores):
        grouped[row["group_id"]].append((row, float(score)))
    detected: dict[tuple[str, str, float], float] = {}
    false_alerts = 0
    for examples in grouped.values():
        last_alert = -math.inf
        for row, score in sorted(examples, key=lambda pair: pair[0]["anchor_unix_s"]):
            anchor = float(row["anchor_unix_s"])
            if score < threshold or anchor - last_alert < suppression_seconds:
                continue
            last_alert = anchor
            key = event_key(row)
            if row["label"] and key in events:
                detected[key] = max(detected.get(key, 0.0), float(row["event_unix_s"]) - anchor)
            elif row.get("source_cohort") == "control_sample":
                false_alerts += 1
    return {
        "threshold": threshold,
        "suppression_seconds": suppression_seconds,
        "event_count": len(events),
        "detected_events": len(detected),
        "event_recall": len(detected) / len(events) if events else None,
        "warning_seconds": sorted(detected.values()),
        "control_observed_airborne_hours": exposure,
        "control_false_alerts": false_alerts,
        "control_false_alerts_per_1000_hours": false_alerts * 1000 / exposure if exposure else None,
        "warning": "Rate denominator is eligible observed airborne time in sampled controls; sparse development exposure does not establish an operational rate.",
    }


def conservative_threshold(
    rows: Sequence[dict[str, Any]], scores: Sequence[float], alerts_per_1000_hours: float = 1.0,
) -> float:
    controls = sorted((float(score) for row, score in zip(rows, scores)
                       if row.get("source_cohort") == "control_sample"), reverse=True)
    hours = sum(float(row.get("airborne_exposure_seconds", 0.0))
                for row in rows if row.get("source_cohort") == "control_sample") / 3600
    if not controls or hours <= 0:
        raise ValueError("No measured control flight exposure")
    allowed_raw_exceedances = math.floor(hours * alerts_per_1000_hours / 1000)
    if allowed_raw_exceedances >= len(controls):
        return -math.inf
    return math.nextafter(controls[allowed_raw_exceedances], math.inf)
