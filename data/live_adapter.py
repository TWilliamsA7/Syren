import json
import os
import sys
import urllib.request
import time

from detection.engine import DetectionEngine, build_default_engine
from ml.aircraft_warning import AircraftWarningEngine
from shared.flight_state import make_flight_state, validate_flight_state

STALE_AFTER_S = 20
MAX_CONFLICT_FLEET_SIZE = 128
UPDATE_INTERVAL_S = 5.0
LIVE_FETCH_TIMEOUT_S = 4.0

ADSB_LOL_URL = "https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}"
ADSB_FI_URL = "https://opendata.adsb.fi/api/v2/lat/{lat}/lon/{lon}/dist/{radius_nm}"

def parse_aircraft(ac, now_s):
    if ac.get("lat") is None or ac.get("lon") is None:
        return None

    seen_pos = ac.get("seen_pos", 0.0)
    on_ground = ac.get("alt_baro") == "ground"
    
    # Skip over any aircraft that are on the ground
    if on_ground:
        return None

    track = ac.get("track")
    if track is None and on_ground:
        track = ac.get("true_heading")

    return make_flight_state(
        timestamp=round(now_s - seen_pos, 3),
        icao24=ac["hex"],
        flight_id=ac.get("flight"),
        registration=ac.get("r"),
        type_code=ac.get("t"),
        category=ac.get("category"),
        latitude=ac["lat"],
        longitude=ac["lon"],
        altitude_baro_ft=None if on_ground else ac.get("alt_baro"),
        altitude_geom_ft=ac.get("alt_geom"),
        on_ground=on_ground,
        source=ac.get("type"),
        accuracy_m=ac.get("rc") or None,
        stale=seen_pos > STALE_AFTER_S,
        ground_speed_kts=ac.get("gs"),
        track_deg=track,
        vertical_rate_baro_fpm=ac.get("baro_rate"),
        vertical_rate_geom_fpm=ac.get("geom_rate"),
        selected_altitude_ft=ac.get("nav_altitude_mcp"),
        squawk=ac.get("squawk"),
        emergency=ac.get("emergency"),
        seen_age_s=seen_pos,
        origin="live",
    )

def parse_response(response):
    aircraft = response.get("ac")
    if aircraft is None:
        aircraft = response.get("aircraft", [])
    now = response["now"]
    now_s = now / 1000 if now > 1e11 else now

    states = []
    for ac in aircraft:
        state = parse_aircraft(ac, now_s)
        if state is not None:
            states.append(state)
    return states

def fetch_live(api_url, lat, lon, radius_nm, timeout_s=10):
    url = api_url.format(lat=lat, lon=lon, radius_nm=radius_nm)
    request = urllib.request.Request(url, headers={"User-Agent": "syren-hackathon"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.load(response)
    return parse_response(data)


def enrich_snapshot(
    states,
    engine,
    detection_cache,
    timestamp_cache,
    warning_engine=None,
    prediction_cache=None,
    prediction_timestamp_cache=None,
):
    """Attach detector and per-aircraft warning results to a live snapshot.

    ADS-B services can repeat a slightly older position for a track, so those
    samples reuse their last result instead of moving either streaming engine
    back in time. Keep the pairwise conflict scan to bounded snapshots: it
    compares every pair and is too expensive for the full nationwide live feed.
    """
    warning_engine = warning_engine or AircraftWarningEngine()
    prediction_cache = prediction_cache if prediction_cache is not None else {}
    prediction_timestamp_cache = (
        prediction_timestamp_cache if prediction_timestamp_cache is not None else {}
    )
    current = []
    for state in states:
        icao24 = state["icao24"].strip().lower()
        if state["timestamp"] >= timestamp_cache.get(icao24, float("-inf")):
            current.append(state)

    if current:
        if len(current) <= MAX_CONFLICT_FLEET_SIZE:
            results = engine.update_fleet(current)
        else:
            results = [engine.update(state) for state in current]
        for state, result in zip(current, results):
            icao24 = state["icao24"].strip().lower()
            detection_cache[icao24] = result.to_mapping()
            timestamp_cache[icao24] = state["timestamp"]

    for state in states:
        icao24 = state["icao24"].strip().lower()
        if state["timestamp"] <= prediction_timestamp_cache.get(icao24, float("-inf")):
            continue
        prediction_cache[icao24] = warning_engine.update(state).to_mapping()
        prediction_timestamp_cache[icao24] = state["timestamp"]

    return [
        {
            **state,
            "detection": detection_cache.get(state["icao24"].strip().lower(), {
                "icao24": state["icao24"].strip().lower(),
                "flight_id": state.get("flight_id") or state["icao24"],
                "timestamp": state["timestamp"],
                "risk_score": 0.0,
                "severity": "normal",
                "anomalies": [],
            }),
            "prediction": prediction_cache.get(state["icao24"].strip().lower(), {
                "icao24": state["icao24"].strip().lower(),
                "flight_id": state.get("flight_id") or state["icao24"],
                "timestamp": state["timestamp"],
                "status": "collecting_history",
                "evaluated": False,
                "alert": False,
                "signals": [],
            }),
        }
        for state in states
    ]

if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "public/data.jsonl"
    print(f"Starting continuous live feed loop targeting: {out_path} (every 5s)...")

    engine: DetectionEngine = build_default_engine()
    warning_engine = AircraftWarningEngine()
    detection_cache = {}
    timestamp_cache = {}
    prediction_cache = {}
    prediction_timestamp_cache = {}
    try:
        while True:
            cycle_start = time.perf_counter()
            fetch_seconds = validation_seconds = scoring_seconds = write_seconds = 0.0
            aircraft_count = 0
            try:
                stage_start = time.perf_counter()
                states = fetch_live(
                    ADSB_LOL_URL, 38.0, -96.0, 1450, timeout_s=LIVE_FETCH_TIMEOUT_S
                )
                fetch_seconds = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                valid_states = [state for state in states if not validate_flight_state(state)]
                validation_seconds = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                enriched_states = enrich_snapshot(
                    valid_states,
                    engine,
                    detection_cache,
                    timestamp_cache,
                    warning_engine,
                    prediction_cache,
                    prediction_timestamp_cache,
                )
                scoring_seconds = time.perf_counter() - stage_start
                aircraft_count = len(enriched_states)

                stage_start = time.perf_counter()
                temporary_path = out_path + ".tmp"
                with open(temporary_path, "w", encoding="utf-8", newline="\n") as out:
                    for state in enriched_states:
                        out.write(json.dumps(state) + "\n")
                for attempt in range(5):
                    try:
                        os.replace(temporary_path, out_path)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(0.05)
                write_seconds = time.perf_counter() - stage_start
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S Fehler')}] Error fetching/writing live data: {e}", file=sys.stderr)

            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(0.0, UPDATE_INTERVAL_S - elapsed)
            print(
                f"[{time.strftime('%H:%M:%S')}] Live cycle: {aircraft_count} aircraft; "
                f"fetch={fetch_seconds:.3f}s validate={validation_seconds:.3f}s "
                f"score={scoring_seconds:.3f}s write={write_seconds:.3f}s "
                f"cycle={elapsed:.3f}s sleep={sleep_time:.3f}s"
            )
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nLive feed loop stopped by user.")
