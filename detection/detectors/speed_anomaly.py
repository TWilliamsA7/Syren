"""Temporal rule for persistent ground-speed decay."""

from detection.models import Anomaly, FlightState
from detection.detectors.base import RuleDetector, bounded_severity
from detection.temporal.history import FlightHistory


class SpeedAnomalyDetector(RuleDetector):
    def __init__(self, window_seconds: float = 60.0, warning_drop_kts: float = 40.0, critical_drop_kts: float = 80.0) -> None:
        if window_seconds <= 0 or warning_drop_kts <= 0 or critical_drop_kts <= warning_drop_kts:
            raise ValueError("require positive window/drop thresholds and critical > warning")
        self.window_seconds = window_seconds
        self.warning_drop_kts = warning_drop_kts
        self.critical_drop_kts = critical_drop_kts
        self._history: FlightHistory[float] = FlightHistory(window_seconds)

    def update(self, state: FlightState) -> Anomaly | None:
        speed = state.kinematics.ground_speed_kts
        if speed is None or speed < 0 or state.position.stale is True:
            return None
        self._history.append(state.icao24, state.timestamp, speed)
        samples = self._history.get(state.icao24)
        if len(samples) < 2:
            return None
        drop = samples[0].value - samples[-1].value
        if drop < self.warning_drop_kts:
            return None
        confidence = 0.60 + 0.39 * (drop - self.warning_drop_kts) / (self.critical_drop_kts - self.warning_drop_kts)
        label = "critical" if drop >= self.critical_drop_kts else "warning"
        return Anomaly("SPEED_ANOMALY", bounded_severity(confidence), f"{label.title()} ground-speed decay {drop:.0f} kt over {samples[-1].timestamp - samples[0].timestamp:.0f} s")
