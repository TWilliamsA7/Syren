"""Rule for repeated altitude reversals over a bounded recent window."""

from detection.models import Anomaly, FlightState
from detection.detectors.base import RuleDetector, bounded_severity
from detection.temporal.history import FlightHistory
from detection.temporal.trend_analysis import reversal_count, range_of


class AltitudeAnomalyDetector(RuleDetector):
    def __init__(self, window_seconds: float = 120.0, minimum_range_ft: float = 1000.0, warning_reversals: int = 3) -> None:
        if window_seconds <= 0 or minimum_range_ft <= 0 or warning_reversals < 2:
            raise ValueError("window/range must be positive and warning_reversals at least 2")
        self.minimum_range_ft = minimum_range_ft
        self.warning_reversals = warning_reversals
        self._history: FlightHistory[float] = FlightHistory(window_seconds)

    def update(self, state: FlightState) -> Anomaly | None:
        altitude = state.position.altitude_baro_ft
        if altitude is None:
            altitude = state.position.altitude_geom_ft
        if altitude is None or state.position.on_ground is True or state.position.stale is True:
            return None
        self._history.append(state.icao24, state.timestamp, altitude)
        samples = self._history.get(state.icao24)
        values = [sample.value for sample in samples]
        span = range_of(values)
        turns = reversal_count(values, tolerance=100.0)
        if span is None or span < self.minimum_range_ft or turns < self.warning_reversals:
            return None
        excess = turns - self.warning_reversals
        confidence = bounded_severity(0.65 + 0.08 * excess + 0.1 * min(1.0, span / (2 * self.minimum_range_ft)))
        return Anomaly("ALTITUDE_ANOMALY", confidence, f"Altitude oscillation: {turns} trend reversals across {span:.0f} ft")

    def reset(self, icao24: str | None = None) -> None:
        if icao24 is None:
            self._history.clear()
        else:
            self._history.discard(icao24)
