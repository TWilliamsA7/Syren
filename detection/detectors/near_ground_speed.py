"""Detect sustained unusually high ground speed near the ground."""

from detection.detectors.base import RuleDetector
from detection.models import Anomaly, FlightState


class NearGroundSpeedDetector(RuleDetector):
    def __init__(
        self,
        altitude_limit_ft: float = 10_000.0,
        speed_threshold_kts: float = 330.0,
        persistence_s: float = 15.0,
    ) -> None:
        if altitude_limit_ft <= 0 or speed_threshold_kts <= 0 or persistence_s <= 0:
            raise ValueError("altitude, speed, and persistence thresholds must be positive")
        self.altitude_limit_ft = altitude_limit_ft
        self.speed_threshold_kts = speed_threshold_kts
        self.persistence_s = persistence_s
        self._since: dict[str, float] = {}

    def update(self, state: FlightState) -> Anomaly | None:
        altitude = state.position.altitude_baro_ft
        if altitude is None:
            altitude = state.position.altitude_geom_ft
        speed = state.kinematics.ground_speed_kts
        if (
            altitude is None
            or speed is None
            or state.position.on_ground is True
            or state.position.stale is True
            or altitude > self.altitude_limit_ft
            or speed <= self.speed_threshold_kts
        ):
            self._since.pop(state.icao24, None)
            return None
        since = self._since.setdefault(state.icao24, state.timestamp)
        if state.timestamp - since < self.persistence_s:
            return None
        severity = min(0.95, 0.65 + 0.0035 * (speed - self.speed_threshold_kts))
        return Anomaly(
            "AGGRESSIVE_NEAR_GROUND_SPEED",
            severity,
            f"Ground speed {speed:.0f} kt below {self.altitude_limit_ft:.0f} ft for at least {self.persistence_s:.0f} s",
        )

    def reset(self, icao24: str | None = None) -> None:
        if icao24 is None:
            self._since.clear()
        else:
            self._since.pop(icao24, None)
