import json
import sys
import urllib.request

from shared.flight_state import make_flight_state, validate_flight_state

STALE_AFTER_S = 20

ADSB_LOL_URL = "https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}"
ADSB_FI_URL = "https://opendata.adsb.fi/api/v2/lat/{lat}/lon/{lon}/dist/{radius_nm}"

def parse_aircraft(ac, now_s):
    if ac.get("lat") is None or ac.get("lon") is None:
        return None

    seen_pos = ac.get("seen_pos", 0.0)
    on_ground = ac.get("alt_baro") == "ground"
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

if __name__ == "__main__":
    states = fetch_live(ADSB_LOL_URL, 28.4, -81.3, 100)
    out_path = sys.argv[1] if len(sys.argv) > 1 else None

    out = open(out_path, "w") if out_path else None
    for state in states:
        errors = validate_flight_state(state)
        if errors:
            print(state["icao24"], errors)
        if out:
            out.write(json.dumps(state) + "\n")
        else:
            print(state["flight_id"], state["position"]["altitude_baro_ft"],
                  state["kinematics"]["ground_speed_kts"])
    if out:
        out.close()
    print(f"{len(states)} FlightStates")