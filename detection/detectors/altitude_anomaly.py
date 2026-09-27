"""Rule for repeated altitude reversals over a bounded recent window."""

from detection.models import Anomaly, FlightState
from detection.detectors.base import RuleDetector, bounded_severity
from detection.temporal.history import FlightHistory
from detection.temporal.trend_analysis import range_of


class AltitudeAnomalyDetector(RuleDetector):
    def __init__(self, window_seconds: float = 120.0, minimum_range_ft: float = 1200.0, warning_reversals: int = 2) -> None:
        if window_seconds <= 0 or minimum_range_ft <= 0 or warning_reversals < 2:
            raise ValueError("window/range must be positive and warning_reversals at least 2")
        self.minimum_range_ft = minimum_range_ft
        self.warning_reversals = warning_reversals
        self._history: FlightHistory[tuple[float, float | None]] = FlightHistory(window_seconds)

    def update(self, state: FlightState) -> Anomaly | None:
        altitude = state.position.altitude_baro_ft
        if altitude is None:
            altitude = state.position.altitude_geom_ft
        if altitude is None or state.position.on_ground is True or state.position.stale is True:
            return None
        rate = state.kinematics.vertical_rate_baro_fpm
        if rate is None:
            rate = state.kinematics.vertical_rate_geom_fpm
        self._history.append(state.icao24, state.timestamp, (altitude, rate))
        samples = self._history.get(state.icao24)
        values = [sample.value[0] for sample in samples]
        span = range_of(values)
        if span is None or span < self.minimum_range_ft:
            return None
        # One-second altitude steps are far smaller than 100 ft even when a
        # meaningful oscillation is underway. Rate sign changes detect that
        # motion; altitude swings preserve support for sparse-rate telemetry.
        rates = [sample.value[1] for sample in samples]
        if any(rate is not None and abs(rate) > 500 for rate in rates):
            turns = self._rate_reversals(rates)
        else:
            turns = self._altitude_reversals(values)
        if turns < self.warning_reversals:
            return None
        excess = turns - self.warning_reversals
        confidence = bounded_severity(0.65 + 0.08 * excess + 0.1 * min(1.0, span / (2 * self.minimum_range_ft)))
        return Anomaly("ALTITUDE_ANOMALY", confidence, f"Altitude oscillation: {turns} trend reversals across {span:.0f} ft")

    @staticmethod
    def _rate_reversals(rates: list[float | None], threshold_fpm: float = 500.0) -> int:
        direction = 0
        turns = 0
        for rate in rates:
            sign = 1 if rate is not None and rate > threshold_fpm else -1 if rate is not None and rate < -threshold_fpm else 0
            if sign:
                if direction and sign != direction:
                    turns += 1
                direction = sign
        return turns

    @staticmethod
    def _altitude_reversals(values: list[float], hysteresis_ft: float = 100.0) -> int:
        if not values:
            return 0
        extreme = values[0]
        direction = 0
        turns = 0
        for value in values[1:]:
            if direction == 0:
                if value >= extreme + hysteresis_ft:
                    direction, extreme = 1, value
                elif value <= extreme - hysteresis_ft:
                    direction, extreme = -1, value
            elif direction > 0:
                if value > extreme:
                    extreme = value
                elif value <= extreme - hysteresis_ft:
                    turns += 1
                    direction, extreme = -1, value
            elif value < extreme:
                extreme = value
            elif value >= extreme + hysteresis_ft:
                turns += 1
                direction, extreme = 1, value
        return turns

    def reset(self, icao24: str | None = None) -> None:
        if icao24 is None:
            self._history.clear()
        else:
            self._history.discard(icao24)
