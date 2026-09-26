"""Detect sustained/repeated loss of recent position reports."""

from detection.detectors.base import RuleDetector
from detection.models import Anomaly, FlightState
from detection.temporal.history import FlightHistory


class TelemetryQualityDetector(RuleDetector):
    def __init__(self, window_seconds: float = 90.0, stale_samples_for_alert: int = 2, stale_age_seconds: float = 20.0) -> None:
        if window_seconds <= 0 or stale_samples_for_alert < 1 or stale_age_seconds <= 0:
            raise ValueError("window/age must be positive and stale sample count at least 1")
        self.stale_samples_for_alert = stale_samples_for_alert
        self.stale_age_seconds = stale_age_seconds
        self._history: FlightHistory[bool] = FlightHistory(window_seconds)

    def update(self, state: FlightState) -> Anomaly | None:
        age = state.status.seen_age_s
        stale = state.position.stale is True or (age is not None and age >= self.stale_age_seconds)
        self._history.append(state.icao24, state.timestamp, stale)
        samples = self._history.get(state.icao24)
        stale_count = sum(sample.value for sample in samples)
        if stale_count < self.stale_samples_for_alert:
            return None
        severity = min(0.90, 0.60 + 0.05 * (stale_count - self.stale_samples_for_alert))
        return Anomaly(
            "TELEMETRY_GAP",
            severity,
            f"Position telemetry stale in {stale_count} of {len(samples)} recent samples",
        )

    def reset(self, icao24: str | None = None) -> None:
        if icao24 is None:
            self._history.clear()
        else:
            self._history.discard(icao24)
