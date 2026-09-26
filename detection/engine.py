"""Detection engine implementing the FlightState -> DetectionResult contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any, Protocol, TypeVar

from detection.aggregation.risk import combine_prototype_risk
from detection.detectors import (
    AltitudeAnomalyDetector,
    AcceleratingDescentDetector,
    AircraftConflictDetector,
    EmergencySquawkDetector,
    HeadingAnomalyDetector,
    HeadingOscillationDetector,
    NearGroundSpeedDetector,
    RapidDescentDetector,
    SpeedAnomalyDetector,
    TelemetryQualityDetector,
    VerticalTelemetryConsistencyDetector,
)
from detection.models import (
    Anomaly,
    DetectionResult,
    FlightState,
    flight_state_from_mapping,
    severity_for_risk,
)


StateT = TypeVar("StateT", contravariant=True)


class Detector(Protocol[StateT]):
    """A detector consumes one state and optionally yields an anomaly."""

    def update(self, state: StateT) -> Anomaly | None:
        """Evaluate a state, updating any detector-owned temporal state."""
        ...


class DetectionEngine:
    """Validate input, run per-aircraft rules, aggregate risk, and emit results.

    Streaming input must be chronological for each aircraft. Use
    ``update_fleet`` once per synchronized snapshot to enable pair conflicts.
    """

    def __init__(
        self,
        detectors: Sequence[Detector[FlightState]] | None = None,
        fleet_detector: AircraftConflictDetector | None = None,
    ) -> None:
        if detectors is None:
            self._detectors = self._default_detectors()
            self._fleet_detector = fleet_detector or AircraftConflictDetector()
        else:
            self._detectors = tuple(detectors)
            self._fleet_detector = fleet_detector
        self._last_timestamp: dict[str, float] = {}

    def update(self, state: FlightState | Mapping[str, Any]) -> DetectionResult:
        """Accept one FlightState or protocol mapping and return its result."""
        parsed = self._coerce_state(state)
        self._validate_order(parsed)
        return self._update_validated(parsed)

    def update_mapping(self, state: Mapping[str, Any]) -> dict[str, object]:
        """Convenience boundary for callers that need a JSON-ready result."""
        return self.update(state).to_mapping()

    def update_fleet(
        self, states: Sequence[FlightState | Mapping[str, Any]]
    ) -> tuple[DetectionResult, ...]:
        """Process one synchronized fleet snapshot and attach conflict alerts.

        A snapshot may contain at most one state per ICAO24. Input validation is
        completed before detector history is mutated, so a malformed snapshot
        cannot partially advance the engine.
        """
        parsed = tuple(self._coerce_state(state) for state in states)
        identifiers = [state.icao24 for state in parsed]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("fleet snapshot contains duplicate icao24 values")
        for state in parsed:
            self._validate_order(state)

        results = [self._update_validated(state) for state in parsed]
        if self._fleet_detector is None or not results:
            return tuple(results)
        conflicts = self._fleet_detector.evaluate_fleet(parsed)
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

    def reset(self, icao24: str | None = None) -> None:
        """Clear all history, or clear history for one aircraft before replay."""
        normalized_id = icao24.strip().lower() if icao24 is not None else None
        for detector in self._detectors:
            reset = getattr(detector, "reset", None)
            if reset is not None:
                reset(normalized_id)
        if normalized_id is None:
            self._last_timestamp.clear()
        else:
            self._last_timestamp.pop(normalized_id, None)

    @staticmethod
    def _coerce_state(state: FlightState | Mapping[str, Any]) -> FlightState:
        if isinstance(state, FlightState):
            parsed = state
        elif isinstance(state, Mapping):
            parsed = flight_state_from_mapping(state)
        else:
            raise TypeError("state must be FlightState or a protocol-shaped mapping")
        if isinstance(parsed.timestamp, bool) or not isinstance(parsed.timestamp, (int, float)) or not isfinite(parsed.timestamp):
            raise ValueError("timestamp must be finite numeric Unix seconds")
        if not isinstance(parsed.icao24, str) or not parsed.icao24.strip():
            raise ValueError("icao24 must be a non-empty string")
        if not isinstance(parsed.flight_id, str):
            raise ValueError("flight_id must be a string")
        return replace(
            parsed,
            icao24=parsed.icao24.strip().lower(),
            flight_id="".join(parsed.flight_id.split()) or parsed.icao24.strip().lower(),
            timestamp=float(parsed.timestamp),
        )

    def _validate_order(self, state: FlightState) -> None:
        previous = self._last_timestamp.get(state.icao24)
        if previous is not None and state.timestamp < previous:
            raise ValueError(
                f"out-of-order state for {state.icao24}: {state.timestamp} < {previous}; "
                "reset the aircraft history before replaying older data"
            )

    def _update_validated(self, state: FlightState) -> DetectionResult:
        anomalies = tuple(
            anomaly
            for detector in self._detectors
            if (anomaly := detector.update(state)) is not None
        )
        risk = combine_prototype_risk(anomalies)
        self._last_timestamp[state.icao24] = state.timestamp
        return DetectionResult(
            icao24=state.icao24,
            flight_id=state.flight_id or state.icao24,
            timestamp=state.timestamp,
            risk_score=risk,
            severity=severity_for_risk(risk),
            anomalies=anomalies,
        )

    @staticmethod
    def _default_detectors() -> tuple[Detector[FlightState], ...]:
        return (
            RapidDescentDetector(),
            AcceleratingDescentDetector(),
            SpeedAnomalyDetector(),
            NearGroundSpeedDetector(),
            HeadingAnomalyDetector(),
            HeadingOscillationDetector(),
            AltitudeAnomalyDetector(),
            EmergencySquawkDetector(),
            TelemetryQualityDetector(),
            VerticalTelemetryConsistencyDetector(),
        )


def build_default_engine() -> DetectionEngine:
    """Create the initial rule-based detector set with prototype thresholds."""
    return DetectionEngine()
