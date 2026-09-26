import math 
from dataclasses import dataclass 

from shared.flight_state import (
    make_flight_state, round_altitude, round_track, round_vertical_rate,
)

NM_PER_DEG_LAT = 60.0
ACCEL_KTS_PER_S = 1.5
TURN_RATE_DEG_PER_S = 3.0
VS_CHANGE_FPM_PER_S = 200.0
ALT_CAPTURE_FPM_PER_FT = 4.0

def _approach(current, target, max_change):
    if target > current:
        return min(current + max_change, target)
    return max(current - max_change, target)


@dataclass
class SimAircraft:
    icao24: str
    callsign: str
    type_code: str
    category: str
    latitude: float
    longitude: float
    altitude_ft: float
    speed_kts: float
    track_deg: float
    vertical_rate_fpm: float = 0.0
    selected_altitude_ft: float | None = None
    registration: str | None = None
    squawk: str = "1200"
    emergency: str = "none"
    on_ground: bool = False
    target_speed_kts: float | None = None
    target_track_deg: float | None = None
    target_vertical_rate_fpm: float | None = None
    climb_rate_fpm: float = 2000.0
    descent_rate_fpm: float = 2000.0
    transponder_on: bool = True


    def _desired_vertical_rate(self):
        if self.target_vertical_rate_fpm is not None:
            return self.target_vertical_rate_fpm
        if self.selected_altitude_ft is None:
            return None
        error_ft = self.selected_altitude_ft - self.altitude_ft
        desired = error_ft * ALT_CAPTURE_FPM_PER_FT
        return max(-self.descent_rate_fpm, min(self.climb_rate_fpm, desired))

    def _update_controls(self, dt):
        if self.target_speed_kts is not None:
            self.speed_kts = _approach(
                self.speed_kts, self.target_speed_kts, ACCEL_KTS_PER_S * dt)

        if self.target_track_deg is not None:
            turn = (self.target_track_deg - self.track_deg + 180) % 360 - 180
            max_turn = TURN_RATE_DEG_PER_S * dt
            turn = max(-max_turn, min(max_turn, turn))
            self.track_deg = (self.track_deg + turn) % 360

        desired_vs = self._desired_vertical_rate()
        if desired_vs is not None:
            self.vertical_rate_fpm = _approach(
                self.vertical_rate_fpm, desired_vs, VS_CHANGE_FPM_PER_S * dt)


    def step(self, dt):
        self._update_controls(dt)
        distance_nm = self.speed_kts * dt / 3600
        track_rad = math.radians(self.track_deg)
        north_nm = distance_nm * math.cos(track_rad)
        east_nm = distance_nm * math.sin(track_rad)

        self.latitude += north_nm / NM_PER_DEG_LAT
        self.longitude += east_nm / (NM_PER_DEG_LAT * math.cos(math.radians(self.latitude)))
        self.altitude_ft += self.vertical_rate_fpm * dt / 60

    def to_flight_state(self, timestamp):
        return make_flight_state(
            timestamp=round(timestamp, 3),
            icao24=self.icao24,
            flight_id=self.callsign,
            registration=self.registration,
            type_code=self.type_code,
            category=self.category,
            latitude=round(self.latitude, 6),
            longitude=round(self.longitude, 6),
            altitude_baro_ft=None if self.on_ground else round_altitude(self.altitude_ft),
            on_ground=self.on_ground,
            source="adsb_icao",
            accuracy_m=186,
            ground_speed_kts=round(self.speed_kts, 1),
            track_deg=round_track(self.track_deg),
            vertical_rate_baro_fpm=round_vertical_rate(self.vertical_rate_fpm),
            selected_altitude_ft=self.selected_altitude_ft,
            squawk=self.squawk,
            emergency=self.emergency,
            seen_age_s=0.0,
            origin="sim",
        )