
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

CENTER = (38.0, -96.0)  # same area as the live feed in data/live_adapter.py: the whole US
RADIUS_NM = 1450
PROGRESS_EVERY_FILES = 500  # how often export_day reports progress, out of ~80k trace files

# bits of point[6]
STALE = 1      # no position for 20+ s before this point
NEW_LEG = 2    # a new flight starts here
GEOM_RATE = 4  # point[7] is the GPS vertical rate, not baro
GEOM_ALT = 8   # point[3] is GPS altitude, not baro


# open() for .jsonl and gzip.open() for .jsonl.gz, so callers don't care which it is.
# Level 1 compression is several times faster to write than the default, for a slightly bigger file.
def open_jsonl(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, compresslevel=1)
    return open(path, mode)


# Every point in one trace_full file as a FlightState, oldest first, or only the points where
# keep(timestamp, lat, lon) is true. Callsign, squawk etc. only come every few points in
# point[8], so the last ones are reused, even when they came on a point that isn't kept.
def parse_trace(path, keep=None):
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
        timestamp = round(trace["timestamp"] + seconds, 3)
        if keep is not None and not keep(timestamp, lat, lon):
            continue

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
            timestamp=timestamp,
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


# Great-circle distance in nautical miles, like the radius of adsb.lol's live query.
def _distance_nm(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 3440.065 * math.asin(math.sqrt(a))


# One trace file's valid states inside the time window and the circle, as
# (timestamp, icao24, JSON line), plus how many were dropped as invalid. Runs in a worker
# process, which also turns the states into JSON so the main process doesn't have to.
def _lines_in_area(path, start_s, end_s, lat, lon, radius_nm):
    def keep(timestamp, point_lat, point_lon):
        return (start_s <= timestamp < end_s
                and _distance_nm(lat, lon, point_lat, point_lon) <= radius_nm)

    kept, dropped = [], 0
    for state in parse_trace(path, keep):
        if validate_flight_state(state):
            dropped += 1
        else:
            kept.append((state["timestamp"], state["icao24"], json.dumps(state) + "\n"))
    return kept, dropped


# Write every aircraft's states inside the circle from start ("HH:MM" UTC) for hours
# into out_path, sorted by time. The day must be on disk. Returns (states, aircraft) counts.
# Reads all ~80k trace files of the day, spread over every CPU core, calling
# on_progress(files_done, files_total) along the way if given.
def export_day(date, out_path, start="16:00", hours=1.0, lat=CENTER[0], lon=CENTER[1],
               radius_nm=RADIUS_NM, root=ARCHIVE_DIR, on_progress=None, on_progress=None):
    folder = day_dir(date, root)
    if folder is None:
        raise LookupError(f"{date} is not on disk, get it with: python3 -m data.history swap {date}")
    if not 0 < hours <= 24:
        raise ValueError(f"hours should be between 0 and 24, got {hours}")
    midnight = datetime.datetime.fromisoformat(date).replace(tzinfo=datetime.timezone.utc)
    start_time = datetime.time.fromisoformat(start)
    start_s = midnight.timestamp() + start_time.hour * 3600 + start_time.minute * 60
    in_area = functools.partial(_lines_in_area, start_s=start_s, end_s=start_s + hours * 3600,
                                lat=lat, lon=lon, radius_nm=radius_nm)

    files = glob.glob(os.path.join(folder, "traces", "*", "trace_full_*.json"))
    rows, dropped = [], 0
    if on_progress is not None:
        on_progress(0, len(files))
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for done, (kept, bad) in enumerate(pool.map(in_area, files, chunksize=200), 1):
            rows.extend(kept)
            dropped += bad
            if on_progress is not None and (done % PROGRESS_EVERY_FILES == 0 or done == len(files)):
                on_progress(done, len(files))
    rows.sort(key=lambda row: row[0])

    folder, name = os.path.split(out_path)
    temporary = os.path.join(folder, "partial_" + name)  # renamed once complete, so a killed
    os.makedirs(folder or ".", exist_ok=True)              # export never looks finished
    with open_jsonl(temporary, "wt") as out:
        out.writelines(line for _, _, line in rows)
    os.replace(temporary, out_path)
    aircraft = len({icao24 for _, icao24, _ in rows})
    print(f"{len(rows)} states from {aircraft} aircraft -> {out_path} ({dropped} invalid dropped)")
    return len(rows), aircraft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a history day into a FlightState JSONL file.")
    parser.add_argument("date", help="YYYY-MM-DD, must be on disk (see python3 -m data.history list)")
    parser.add_argument("out", help="output file, .jsonl or .jsonl.gz")
    parser.add_argument("--start", default="16:00", help="UTC start time, HH:MM (default 16:00)")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--lat", type=float, default=CENTER[0])
    parser.add_argument("--lon", type=float, default=CENTER[1])
    parser.add_argument("--radius", type=float, default=RADIUS_NM, help="nautical miles")
    args = parser.parse_args()
    export_day(args.date, args.out, args.start, args.hours, args.lat, args.lon, args.radius)
