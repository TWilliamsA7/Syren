"""Short-horizon aircraft conflict screening from broadcast motion states.

This is a local flat-earth constant-velocity estimate, suitable as a prototype
screening rule rather than a validated separation assurance system.
"""

from __future__ import annotations

from math import cos, radians, sin
from collections.abc import Sequence

from detection.models import Anomaly, FlightState


class AircraftConflictDetector:
    def __init__(
        self,
        lookahead_seconds: float = 120.0,
        horizontal_threshold_nm: float = 5.0,
        vertical_threshold_ft: float = 1000.0,
        step_seconds: float = 5.0,
    ) -> None:
        if min(lookahead_seconds, horizontal_threshold_nm, vertical_threshold_ft, step_seconds) <= 0:
            raise ValueError("conflict prediction settings must be positive")
        self.lookahead_seconds = lookahead_seconds
        self.horizontal_threshold_nm = horizontal_threshold_nm
        self.vertical_threshold_ft = vertical_threshold_ft
        self.step_seconds = step_seconds

    def evaluate_fleet(self, states: Sequence[FlightState]) -> dict[str, tuple[Anomaly, ...]]:
        """Return predicted conflicts grouped by ICAO24 for the current fleet snapshot."""
        usable = [state for state in states if self._usable(state)]
        conflicts: dict[str, list[Anomaly]] = {}
        for index, first in enumerate(usable):
            for second in usable[index + 1 :]:
                if abs(first.timestamp - second.timestamp) > self.step_seconds:
                    continue
                closest = self._closest_approach(first, second)
                if closest is None:
                    continue
                time_s, horizontal_nm, vertical_ft = closest
                accuracy_nm = sum(
                    (state.position.accuracy_m or 0.0) / 1852.0 for state in (first, second)
                )
                horizontal_limit_nm = self.horizontal_threshold_nm + accuracy_nm
                if horizontal_nm > horizontal_limit_nm or vertical_ft > self.vertical_threshold_ft:
                    continue
                severity = min(
                    0.99,
                    0.70
                    + 0.15 * (1.0 - horizontal_nm / horizontal_limit_nm)
                    + 0.14 * (1.0 - vertical_ft / self.vertical_threshold_ft),
                )
                message = (
                    f"Projected proximity in {time_s:.0f} s: "
                    f"{horizontal_nm:.1f} NM horizontal, {vertical_ft:.0f} ft vertical"
                )
                for state, other in ((first, second), (second, first)):
                    conflicts.setdefault(state.icao24, []).append(
                        Anomaly(
                            "AIRCRAFT_CONFLICT",
                            severity,
                            f"Potential conflict with {other.icao24}: {message}",
                        )
                    )
        return {icao24: tuple(items) for icao24, items in conflicts.items()}

    @staticmethod
    def _usable(state: FlightState) -> bool:
        return (
            state.position.latitude is not None
            and -90.0 <= state.position.latitude <= 90.0
            and state.position.longitude is not None
            and -180.0 <= state.position.longitude <= 180.0
            and state.position.stale is not True
            and state.position.on_ground is not True
            and state.kinematics.ground_speed_kts is not None
            and state.kinematics.ground_speed_kts >= 0.0
            and state.kinematics.track_deg is not None
            and 0.0 <= state.kinematics.track_deg < 360.0
            and (state.position.altitude_baro_ft is not None or state.position.altitude_geom_ft is not None)
        )

    def _closest_approach(
        self, first: FlightState, second: FlightState
    ) -> tuple[float, float, float] | None:
        # Use one altitude reference consistently; never compare barometric to geometric altitude.
        if first.position.altitude_baro_ft is not None and second.position.altitude_baro_ft is not None:
            first_alt = first.position.altitude_baro_ft
            second_alt = second.position.altitude_baro_ft
            first_rate = first.kinematics.vertical_rate_baro_fpm
            second_rate = second.kinematics.vertical_rate_baro_fpm
        elif first.position.altitude_geom_ft is not None and second.position.altitude_geom_ft is not None:
            first_alt = first.position.altitude_geom_ft
            second_alt = second.position.altitude_geom_ft
            first_rate = first.kinematics.vertical_rate_geom_fpm
            second_rate = second.kinematics.vertical_rate_geom_fpm
        else:
            return None

        latitude_mid = (first.position.latitude + second.position.latitude) / 2.0
        first_track = radians(first.kinematics.track_deg)
        second_track = radians(second.kinematics.track_deg)
        first_speed = first.kinematics.ground_speed_kts / 3600.0
        second_speed = second.kinematics.ground_speed_kts / 3600.0
        first_vx = first_speed * sin(first_track)
        second_vx = second_speed * sin(second_track)
        first_vy = first_speed * cos(first_track)
        second_vy = second_speed * cos(second_track)
        reference_time = max(first.timestamp, second.timestamp)
        first_age = reference_time - first.timestamp
        second_age = reference_time - second.timestamp
        dx_nm = (second.position.longitude - first.position.longitude) * 60.0 * cos(radians(latitude_mid))
        dy_nm = (second.position.latitude - first.position.latitude) * 60.0
        # Align asynchronous reports to the newer report timestamp before projecting forward.
        dx_nm += second_vx * second_age - first_vx * first_age
        dy_nm += second_vy * second_age - first_vy * first_age
        dvx_nm_s = second_vx - first_vx
        dvy_nm_s = second_vy - first_vy

        # Missing vertical rate means the projection holds current altitude constant;
        # it is a forecast assumption, not a synthesized telemetry value.
        first_vz = (first_rate or 0.0) / 60.0
        second_vz = (second_rate or 0.0) / 60.0
        dvz_ft_s = second_vz - first_vz
        initial_dz_ft = second_alt + second_vz * second_age - first_alt - first_vz * first_age
        best: tuple[float, float, float] | None = None
        steps = max(1, int(self.lookahead_seconds / self.step_seconds))
        for index in range(steps + 1):
            time_s = min(index * self.step_seconds, self.lookahead_seconds)
            horizontal_nm = ((dx_nm + dvx_nm_s * time_s) ** 2 + (dy_nm + dvy_nm_s * time_s) ** 2) ** 0.5
            vertical_ft = abs(initial_dz_ft + dvz_ft_s * time_s)
            if vertical_ft <= self.vertical_threshold_ft and (best is None or horizontal_nm < best[1]):
                best = (time_s, horizontal_nm, vertical_ft)
        return best
