
import argparse
import concurrent.futures
import datetime
import functools
import glob
import gzip
import json
import math
import os

from data.history import ARCHIVE_DIR, day_dir
from shared.flight_state import make_flight_state, round_track, validate_flight_state

ORLANDO = (28.4, -81.3)  # same area as the live feed
RADIUS_NM = 100

# bits of point[6]
STALE = 1      # no position for 20+ s before this point
NEW_LEG = 2    # a new flight starts here
GEOM_RATE = 4  # point[7] is the GPS vertical rate, not baro
GEOM_ALT = 8   # point[3] is GPS altitude, not baro


# open() for .jsonl and gzip.open() for .jsonl.gz, so callers don't care which it is.
def open_jsonl(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


# Every point in one trace_full file as a FlightState, oldest first.
# Callsign, squawk etc. only come every few points in point[8], so the last ones are reused.
def parse_trace(path):
    with gzip.open(path) as f:
        trace = json.load(f)
    details = {}
    states = []
    for point in trace["trace"]:
        point = point + [None] * (12 - len(point))
        seconds, lat, lon, altitude, speed, track, flags, rate, new_details, source = point[:10]
        altitude_geom, rate_geom = point[10], point[11]

        if flags & NEW_LEG:
            details = {}  # a new flight: don't carry the last one's callsign or squawk
        if new_details:
            details = {**details, **new_details}

        on_ground = altitude == "ground"
        altitude_baro = None if on_ground else altitude
        if flags & GEOM_ALT:
            altitude_baro = None
            if altitude_geom is None:
                altitude_geom = altitude
        rate_baro = rate
        if flags & GEOM_RATE:
            rate_baro = None
            if rate_geom is None:
                rate_geom = rate

        states.append(make_flight_state(
            timestamp=round(trace["timestamp"] + seconds, 3),
            icao24=trace["icao"],
            flight_id=details.get("flight"),
            registration=trace.get("r"),
            type_code=trace.get("t"),
            category=details.get("category"),
            latitude=lat,
            longitude=lon,
            altitude_baro_ft=altitude_baro,
            altitude_geom_ft=altitude_geom,
            on_ground=on_ground,
            source=source or details.get("type") or "unknown",
            accuracy_m=details.get("rc") or None,  # rc 0 means unknown
            stale=bool(flags & STALE),
            ground_speed_kts=speed,
            track_deg=round_track(track),
            vertical_rate_baro_fpm=rate_baro,
            vertical_rate_geom_fpm=rate_geom,
            selected_altitude_ft=details.get("nav_altitude_mcp"),
            squawk=details.get("squawk"),
            emergency=details.get("emergency"),
            origin="history",
        ))
    return states


# Rough distance in nautical miles, fine for a few hundred miles.
def _distance_nm(lat1, lon1, lat2, lon2):
    north = (lat2 - lat1) * 60
    east = (lon2 - lon1) * 60 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(north, east)


# One trace file's valid states inside the time window and the circle, plus how many
# states were dropped as invalid. Runs in a worker process.
def _states_in_area(path, start_s, end_s, lat, lon, radius_nm):
    kept, dropped = [], 0
    for state in parse_trace(path):
        position = state["position"]
        if not start_s <= state["timestamp"] < end_s:
            continue
        if _distance_nm(lat, lon, position["latitude"], position["longitude"]) > radius_nm:
            continue
        if validate_flight_state(state):
            dropped += 1
        else:
            kept.append(state)
    return kept, dropped


# Write every aircraft's states inside the circle from start ("HH:MM" UTC) for hours
# into out_path, sorted by time. The day must be on disk. Returns (states, aircraft) counts.
# Reads all ~80k trace files of the day, spread over every CPU core.
def export_day(date, out_path, start="16:00", hours=1.0, lat=ORLANDO[0], lon=ORLANDO[1],
               radius_nm=RADIUS_NM, root=ARCHIVE_DIR):
    folder = day_dir(date, root)
    if folder is None:
        raise LookupError(f"{date} is not on disk, get it with: python3 -m data.history swap {date}")
    if not 0 < hours <= 24:
        raise ValueError(f"hours should be between 0 and 24, got {hours}")
    midnight = datetime.datetime.fromisoformat(date).replace(tzinfo=datetime.timezone.utc)
    start_time = datetime.time.fromisoformat(start)
    start_s = midnight.timestamp() + start_time.hour * 3600 + start_time.minute * 60
    in_area = functools.partial(_states_in_area, start_s=start_s, end_s=start_s + hours * 3600,
                                lat=lat, lon=lon, radius_nm=radius_nm)

    files = glob.glob(os.path.join(folder, "traces", "*", "trace_full_*.json"))
    states, dropped = [], 0
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for kept, bad in pool.map(in_area, files, chunksize=200):
            states.extend(kept)
            dropped += bad
    states.sort(key=lambda state: state["timestamp"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open_jsonl(out_path, "wt") as out:
        for state in states:
            out.write(json.dumps(state) + "\n")
    aircraft = len({state["icao24"] for state in states})
    print(f"{len(states)} states from {aircraft} aircraft -> {out_path} ({dropped} invalid dropped)")
    return len(states), aircraft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a history day into a FlightState JSONL file.")
    parser.add_argument("date", help="YYYY-MM-DD, must be on disk (see python3 -m data.history list)")
    parser.add_argument("out", help="output file, .jsonl or .jsonl.gz")
    parser.add_argument("--start", default="16:00", help="UTC start time, HH:MM (default 16:00)")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--lat", type=float, default=ORLANDO[0])
    parser.add_argument("--lon", type=float, default=ORLANDO[1])
    parser.add_argument("--radius", type=float, default=RADIUS_NM, help="nautical miles")
    args = parser.parse_args()
    export_day(args.date, args.out, args.start, args.hours, args.lat, args.lon, args.radius)