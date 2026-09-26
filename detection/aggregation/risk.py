"""Prototype risk aggregation for DetectionResult anomalies.

The noisy-OR combination is a transparent hackathon heuristic, not a calibrated
probability. Multiple high-severity signals raise risk while the score remains
bounded in [0, 1].
"""

from __future__ import annotations

from collections.abc import Iterable
from detection.models import Anomaly


def combine_prototype_risk(signals: Iterable[Anomaly]) -> float:
    """Combine anomaly severities with noisy-OR; no signals yield zero risk."""
    remaining_risk = 1.0
    for signal in signals:
        if not 0.0 <= signal.severity <= 1.0:
            raise ValueError("anomaly severity must be in [0, 1]")
        remaining_risk *= 1.0 - signal.severity
    return 1.0 - remaining_risk
