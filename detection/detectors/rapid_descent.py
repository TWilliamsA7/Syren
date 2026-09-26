"""Rule for unusually fast barometric or geometric descent."""

from detection.models import Anomaly, FlightState
from detection.detectors.base import RuleDetector, bounded_severity


class RapidDescentDetector(RuleDetector):
    def __init__(self, warning_fpm: float = 3000.0, critical_fpm: float = 5000.0) -> None:
        if warning_fpm <= 0 or critical_fpm <= warning_fpm:
            raise ValueError("critical_fpm must exceed positive warning_fpm")
        self.warning_fpm = warning_fpm
        self.critical_fpm = critical_fpm

    def update(self, state: FlightState) -> Anomaly | None:
        if state.position.on_ground is True or state.position.stale is True:
            return None
        rates = (
            ("baro", state.kinematics.vertical_rate_baro_fpm),
            ("geom", state.kinematics.vertical_rate_geom_fpm),
        )
        valid = [(source, rate) for source, rate in rates if rate is not None and rate < 0]
        if not valid:
            return None
        source, rate = min(valid, key=lambda pair: pair[1])
        descent = -rate
        if descent < self.warning_fpm:
            return None
        confidence = 0.60 + 0.39 * (descent - self.warning_fpm) / (self.critical_fpm - self.warning_fpm)
        label = "critical" if descent >= self.critical_fpm else "warning"
        return Anomaly("RAPID_DESCENT", bounded_severity(confidence), f"{label.title()} descent rate {rate:.0f} fpm ({source})")
