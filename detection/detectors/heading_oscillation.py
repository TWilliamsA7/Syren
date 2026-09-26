"""Detect repeated reversals in course changes over a recent window."""

from detection.detectors.base import RuleDetector
from detection.models import Anomaly, FlightState
from detection.temporal.history import FlightHistory
from detection.temporal.trend_analysis import reversal_count, range_of


class HeadingOscillationDetector(RuleDetector):
    def __init__(self, window_seconds: float = 90.0, minimum_span_deg: float = 50.0, minimum_reversals: int = 3) -> None:
        if window_seconds <= 0 or minimum_span_deg <= 0 or minimum_reversals < 2:
            raise ValueError("window/span must be positive and minimum_reversals at least 2")
        self.minimum_span_deg = minimum_span_deg
        self.minimum_reversals = minimum_reversals
        self._history: FlightHistory[float] = FlightHistory(window_seconds)

    def update(self, state: FlightState) -> Anomaly | None:
        track = state.kinematics.track_deg
        if track is None or not 0 <= track < 360 or state.position.stale is True or state.position.on_ground is True:
            return None
        self._history.append(state.icao24, state.timestamp, track)
        tracks = [sample.value for sample in self._history.get(state.icao24)]
        if len(tracks) < self.minimum_reversals + 2:
            return None
        # Unwrap bearings into cumulative signed changes to handle 359 -> 0.
        changes: list[float] = []
        for previous, current in zip(tracks, tracks[1:]):
            signed_change = (current - previous + 180.0) % 360.0 - 180.0
            changes.append(signed_change)
        span = range_of([0.0, *self._unwrap(tracks)])
        turns = reversal_count(changes, tolerance=3.0)
        if span is None or span < self.minimum_span_deg or turns < self.minimum_reversals:
            return None
        return Anomaly(
            "HEADING_OSCILLATION",
            min(0.90, 0.62 + 0.05 * (turns - self.minimum_reversals)),
            f"Repeated course reversals: {turns} direction changes in a {span:.0f}-degree span",
        )

    def reset(self, icao24: str | None = None) -> None:
        if icao24 is None:
            self._history.clear()
        else:
            self._history.discard(icao24)

    @staticmethod
    def _unwrap(tracks: list[float]) -> list[float]:
        if not tracks:
            return []
        result = [tracks[0]]
        for previous, current in zip(tracks, tracks[1:]):
            result.append(result[-1] + ((current - previous + 180.0) % 360.0 - 180.0))
        return result
