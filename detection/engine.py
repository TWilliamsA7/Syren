"""Detection engine implementing the FlightState -> DetectionResult contract."""

from __future__ import annotations

from typing import Protocol, TypeVar

from detection.aggregation.risk import combine_prototype_risk
from detection.detectors import (
    AltitudeAnomalyDetector,
    EmergencySquawkDetector,
    HeadingAnomalyDetector,
    RapidDescentDetector,
    SpeedAnomalyDetector,
)
from detection.models import Anomaly, DetectionResult, FlightState, severity_for_risk


StateT = TypeVar("StateT", contravariant=True)


class Detector(Protocol[StateT]):
    """A detector consumes one state and optionally yields an anomaly."""

    def update(self, state: StateT) -> Anomaly | None:
        """Evaluate a state, updating any detector-owned temporal state."""
        ...


class DetectionEngine:
    """Run rule detectors and return one result for each input flight state."""

    def __init__(self, detectors: tuple[Detector[FlightState], ...]) -> None:
        self._detectors = detectors

    def update(self, state: FlightState) -> DetectionResult:
        """Evaluate state, aggregate active signals, and produce its result."""
        anomalies = tuple(
            anomaly
            for detector in self._detectors
            if (anomaly := detector.update(state)) is not None
        )
        risk = combine_prototype_risk(anomalies)
        return DetectionResult(
            icao24=state.icao24,
            flight_id=state.flight_id or state.icao24,
            timestamp=state.timestamp,
            risk_score=risk,
            severity=severity_for_risk(risk),
            anomalies=anomalies,
        )


def build_default_engine() -> DetectionEngine:
    """Create the initial rule-based detector set with prototype thresholds."""
    return DetectionEngine((
        RapidDescentDetector(),
        SpeedAnomalyDetector(),
        HeadingAnomalyDetector(),
        AltitudeAnomalyDetector(),
        EmergencySquawkDetector(),
    ))
