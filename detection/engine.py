"""Detection engine implementing the FlightState -> DetectionResult contract."""

from __future__ import annotations

from typing import Protocol, TypeVar
from collections.abc import Sequence

from detection.aggregation.risk import combine_prototype_risk
from detection.detectors import (
    AltitudeAnomalyDetector,
    AcceleratingDescentDetector,
    AircraftConflictDetector,
    EmergencySquawkDetector,
    HeadingAnomalyDetector,
    HeadingOscillationDetector,
    RapidDescentDetector,
    SpeedAnomalyDetector,
    TelemetryQualityDetector,
    VerticalTelemetryConsistencyDetector,
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

    def __init__(
        self,
        detectors: tuple[Detector[FlightState], ...],
        fleet_detector: AircraftConflictDetector | None = None,
    ) -> None:
        self._detectors = detectors
        self._fleet_detector = fleet_detector

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

    def update_fleet(self, states: Sequence[FlightState]) -> tuple[DetectionResult, ...]:
        """Process one fleet snapshot and append any pairwise conflict alerts."""
        results = [self.update(state) for state in states]
        if self._fleet_detector is None or not results:
            return tuple(results)
        conflicts = self._fleet_detector.evaluate_fleet(states)
        for index, result in enumerate(results):
            additional = conflicts.get(result.icao24, ())
            if not additional:
                continue
            anomalies = result.anomalies + additional
            risk = combine_prototype_risk(anomalies)
            results[index] = DetectionResult(
                icao24=result.icao24,
                flight_id=result.flight_id,
                timestamp=result.timestamp,
                risk_score=risk,
                severity=severity_for_risk(risk),
                anomalies=anomalies,
            )
        return tuple(results)


def build_default_engine() -> DetectionEngine:
    """Create the initial rule-based detector set with prototype thresholds."""
    return DetectionEngine((
        RapidDescentDetector(),
        AcceleratingDescentDetector(),
        SpeedAnomalyDetector(),
        HeadingAnomalyDetector(),
        HeadingOscillationDetector(),
        AltitudeAnomalyDetector(),
        EmergencySquawkDetector(),
        TelemetryQualityDetector(),
        VerticalTelemetryConsistencyDetector(),
    ), fleet_detector=AircraftConflictDetector())
