"""Rule for abrupt changes in reported ground track."""

from detection.models import Anomaly, FlightState
from detection.detectors.base import RuleDetector, bounded_severity


def angular_difference_degrees(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


class HeadingAnomalyDetector(RuleDetector):
    def __init__(self, warning_change_deg: float = 45.0, critical_change_deg: float = 120.0) -> None:
        if not 0 < warning_change_deg < critical_change_deg <= 180:
            raise ValueError("require 0 < warning < critical <= 180 degrees")
        self.warning_change_deg = warning_change_deg
        self.critical_change_deg = critical_change_deg
        self._previous: dict[str, tuple[float, float]] = {}

    def update(self, state: FlightState) -> Anomaly | None:
        track = state.kinematics.track_deg
        if (
            track is None
            or not 0 <= track < 360
            or state.position.stale is True
            or state.position.on_ground is True
        ):
            return None
        prior = self._previous.get(state.icao24)
        self._previous[state.icao24] = (state.timestamp, track)
        if prior is None or state.timestamp <= prior[0] or state.timestamp - prior[0] > 30:
            return None
        change = angular_difference_degrees(track, prior[1])
        if change < self.warning_change_deg:
            return None
        confidence = 0.60 + 0.39 * (change - self.warning_change_deg) / (self.critical_change_deg - self.warning_change_deg)
        label = "critical" if change >= self.critical_change_deg else "warning"
        return Anomaly("HEADING_ANOMALY", bounded_severity(confidence), f"{label.title()} track change {change:.0f} degrees")

    def reset(self, icao24: str | None = None) -> None:
        if icao24 is None:
            self._previous.clear()
        else:
            self._previous.pop(icao24, None)
