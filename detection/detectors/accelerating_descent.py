"""Detect a descent rate that is becoming more negative over time."""

from detection.detectors.base import RuleDetector
from detection.models import Anomaly, FlightState
from detection.temporal.history import FlightHistory
from detection.temporal.trend_analysis import linear_slope


class AcceleratingDescentDetector(RuleDetector):
    def __init__(self, window_seconds: float = 30.0, minimum_acceleration_fpm_per_min: float = 600.0) -> None:
        if window_seconds <= 0 or minimum_acceleration_fpm_per_min <= 0:
            raise ValueError("window and acceleration threshold must be positive")
        self.window_seconds = window_seconds
        self.minimum_acceleration = minimum_acceleration_fpm_per_min
        self._history: FlightHistory[float] = FlightHistory(window_seconds)

    def update(self, state: FlightState) -> Anomaly | None:
        if state.position.on_ground is True or state.position.stale is True:
            return None
        rate = state.kinematics.vertical_rate_baro_fpm
        if rate is None:
            rate = state.kinematics.vertical_rate_geom_fpm
        if rate is None:
            return None
        self._history.append(state.icao24, state.timestamp, rate)
        samples = self._history.get(state.icao24)
        if len(samples) < 3 or samples[-1].value >= 0:
            return None
        slope_per_second = linear_slope([(sample.timestamp, sample.value) for sample in samples])
        if slope_per_second is None:
            return None
        acceleration = -slope_per_second * 60.0
        if acceleration < self.minimum_acceleration:
            return None
        severity = min(0.95, 0.60 + 0.35 * acceleration / (2 * self.minimum_acceleration))
        return Anomaly(
            "ACCELERATING_DESCENT",
            severity,
            f"Descent rate is worsening by about {acceleration:.0f} fpm per minute",
        )
