import math
from dataclasses import dataclass

from simulation.aircraft import NM_PER_DEG_LAT, SimAircraft

SPEED_LIMIT_ALT_FT = 10000
APPROACH_ALT_FT = 3000
APPROACH_DIST_NM = 10
DESCENT_NM_PER_1000_FT = 3
MAX_GLIDE_FPM = 1500
TOUCHDOWN_HEIGHT_FT = 50

@dataclass
class Airport:
    code: str
    latitude: float
    longitude: float
    elevation_ft: float


AIRPORTS = {
    "MCO": Airport("MCO", 28.4294, -81.3090, 96),
    "TPA": Airport("TPA", 27.9755, -82.5332, 26),
    "MIA": Airport("MIA", 25.7959, -80.2870, 8),
    "JAX": Airport("JAX", 30.4941, -81.6879, 30),
    "RSW": Airport("RSW", 26.5362, -81.7552, 30),
}

def distance_and_bearing(lat1, lon1, lat2, lon2):
    north_nm = (lat2 - lat1) * NM_PER_DEG_LAT
    mid_lat = math.radians((lat1 + lat2) / 2)
    east_nm = (lon2 - lon1) * NM_PER_DEG_LAT * math.cos(mid_lat)
    distance = math.hypot(north_nm, east_nm)
    bearing = math.degrees(math.atan2(east_nm, north_nm)) % 360
    return distance, bearing

@dataclass
class FlightPlan:
    origin: Airport
    destination: Airport
    cruise_altitude_ft: float
    cruise_speed_kts: float = 450
    phase: str = "takeoff"

    def spawn(self, icao24, callsign, type_code, category="A3"):
        _, bearing = self._to_destination(self.origin.latitude, self.origin.longitude)
        return SimAircraft(
            icao24=icao24, callsign=callsign, type_code=type_code, category=category,
            latitude=self.origin.latitude, longitude=self.origin.longitude,
            altitude_ft=self.origin.elevation_ft + 50,
            speed_kts=160, track_deg=bearing,
            selected_altitude_ft=self.cruise_altitude_ft,
        )

    @property
    def finished(self):
        return self.phase == "landed"

    def update(self, ac):
        distance_nm, bearing = self._to_destination(ac.latitude, ac.longitude)
        self._advance_phase(ac, distance_nm)
        self._set_targets(ac, distance_nm, bearing)

    def _to_destination(self, lat, lon):
        return distance_and_bearing(
            lat, lon, self.destination.latitude, self.destination.longitude)

    def _top_of_descent_nm(self, altitude_ft):
        above_approach = altitude_ft - self.destination.elevation_ft - APPROACH_ALT_FT
        return max(0, above_approach) / 1000 * DESCENT_NM_PER_1000_FT + APPROACH_DIST_NM

    def _advance_phase(self, ac, distance_nm):
        alt = ac.altitude_ft
        if self.phase == "takeoff" and alt >= min(SPEED_LIMIT_ALT_FT, self.cruise_altitude_ft - 50):
            self.phase = "climb"
        if self.phase == "climb" and alt >= self.cruise_altitude_ft - 50:
            self.phase = "cruise"
        if self.phase in ("takeoff", "climb", "cruise") and distance_nm <= self._top_of_descent_nm(alt):
            self.phase = "descent"
        if self.phase == "descent" and distance_nm <= APPROACH_DIST_NM:
            self.phase = "approach"
        if self.phase == "approach" and alt <= self.destination.elevation_ft + TOUCHDOWN_HEIGHT_FT:
            self.phase = "landed"
            ac.on_ground = True
            ac.altitude_ft = self.destination.elevation_ft
            ac.vertical_rate_fpm = 0.0

    def _set_targets(self, ac, distance_nm, bearing):
        if self.phase == "landed":
            ac.selected_altitude_ft = None
            ac.target_vertical_rate_fpm = 0.0
            ac.target_speed_kts = 0.0
            return

        if distance_nm > 1:
            ac.target_track_deg = bearing

        if self.phase == "takeoff":
            ac.climb_rate_fpm = 3000
            ac.target_speed_kts = 250
        elif self.phase in ("climb", "cruise"):
            ac.climb_rate_fpm = 2000
            ac.target_speed_kts = self.cruise_speed_kts
        elif self.phase == "descent":
            ac.selected_altitude_ft = self.destination.elevation_ft + APPROACH_ALT_FT
            ac.descent_rate_fpm = 2000
            ac.target_speed_kts = 300 if ac.altitude_ft > SPEED_LIMIT_ALT_FT else 250
        elif self.phase == "approach":
            ac.target_speed_kts = 150
            height_ft = ac.altitude_ft - self.destination.elevation_ft
            minutes_to_go = max(distance_nm / max(ac.speed_kts, 1) * 60, 0.1)
            ac.target_vertical_rate_fpm = max(-MAX_GLIDE_FPM, -height_ft / minutes_to_go)