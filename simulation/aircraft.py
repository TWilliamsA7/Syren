import math 
from dataclasses import dataclass 

from shared.flight_state import (
    make_flight_state, round_altitude, round_track, round_vertical_rate,
)

NM_PER_DEG_LAT = 60.0

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

    def step(self, dt):
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