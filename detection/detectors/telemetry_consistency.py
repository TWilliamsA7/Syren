"""Compare reported vertical rate with observed barometric-altitude change."""

from detection.detectors.base import RuleDetector
from detection.models import Anomaly, FlightState


class VerticalTelemetryConsistencyDetector(RuleDetector):
    def __init__(self, mismatch_tolerance_ft: float = 700.0, max_interval_seconds: float = 30.0) -> None:
        if mismatch_tolerance_ft <= 0 or max_interval_seconds <= 0:
            raise ValueError("tolerance and max interval must be positive")
        self.mismatch_tolerance_ft = mismatch_tolerance_ft
        self.max_interval_seconds = max_interval_seconds
        self._previous: dict[str, tuple[float, float, float]] = {}

    def update(self, state: FlightState) -> Anomaly | None:
        altitude = state.position.altitude_baro_ft
        rate = state.kinematics.vertical_rate_baro_fpm
        if altitude is None or rate is None or state.position.stale is True or state.position.on_ground is True:
            return None
        previous = self._previous.get(state.icao24)
        self._previous[state.icao24] = (state.timestamp, altitude, rate)
        if previous is None:
            return None
        previous_time, previous_altitude, previous_rate = previous
        elapsed = state.timestamp - previous_time
        if elapsed <= 0 or elapsed > self.max_interval_seconds:
            return None
        expected_change = (previous_rate + rate) * 0.5 * elapsed / 60.0
        actual_change = altitude - previous_altitude
        mismatch = abs(actual_change - expected_change)
        if mismatch < self.mismatch_tolerance_ft:
            return None
        severity = min(0.90, 0.60 + 0.30 * mismatch / (3 * self.mismatch_tolerance_ft))
        return Anomaly(
            "TELEMETRY_INCONSISTENCY",
            severity,
            f"Altitude changed {actual_change:.0f} ft; reported vertical rate implies {expected_change:.0f} ft over {elapsed:.0f} s",
        )
