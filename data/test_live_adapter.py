import sys

from shared.flight_state import validate_flight_state
from data.live_adapter import ADSB_FI_URL, ADSB_LOL_URL, fetch_live, parse_response

AIRCRAFT = {
    "hex": "a5d28c", "type": "adsb_icao", "flight": "UAL3776 ",
    "alt_baro": 14850, "gs": 423.2, "track": 174.17,
    "lat": 28.848358, "lon": -82.270907, "rc": 186, "seen_pos": 0.5,
}


def test_both_response_formats():
    lol = parse_response({"ac": [AIRCRAFT], "now": 1790400021000})
    fi = parse_response({"aircraft": [AIRCRAFT], "now": 1790400021.0})
    assert len(lol) == 1 and len(fi) == 1
    assert lol == fi, "both formats should give identical FlightStates"
    assert lol[0]["timestamp"] == 1790400020.5
    assert validate_flight_state(lol[0]) == []


def test_empty_response():
    assert parse_response({"ac": [], "now": 1790400021000}) == []
    assert parse_response({"now": 1790400021.0}) == []


LIVE_URLS = {
    "adsb.lol": ADSB_LOL_URL,
    "adsb.fi": ADSB_FI_URL,
}


def check_live_urls():
    for name, url in LIVE_URLS.items():
        try:
            states = fetch_live(url, 28.4, -81.3, 100)
        except Exception as e:
            print(f"FAIL {name}: {e}")
            continue
        invalid = [s["icao24"] for s in states if validate_flight_state(s)]
        status = "ok  " if states and not invalid else "FAIL"
        print(f"{status} {name}: {len(states)} aircraft, {len(invalid)} invalid {invalid}")


if __name__ == "__main__":
    test_both_response_formats()
    test_empty_response()
    print("offline tests passed")
    if "--live" in sys.argv:
        check_live_urls()