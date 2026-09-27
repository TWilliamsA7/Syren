import sys

from shared.flight_state import validate_flight_state
from data.live_adapter import ADSB_FI_URL, ADSB_LOL_URL, enrich_snapshot, fetch_live, parse_response
from detection.engine import build_default_engine

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


def test_live_snapshot_contains_frontend_detection_payload():
    aircraft = {**AIRCRAFT, "squawk": "7700", "emergency": "none"}
    states = parse_response({"ac": [aircraft], "now": 1790400021000})
    enriched = enrich_snapshot(states, build_default_engine(), {}, {})
    detection = enriched[0]["detection"]
    assert any(item["type"] == "EMERGENCY_SQUAWK" for item in detection["anomalies"])
    assert detection["severity"] == "critical"


def test_live_snapshot_ignores_out_of_order_positions():
    engine = build_default_engine()
    detection_cache = {}
    timestamp_cache = {}
    newest = parse_response({"ac": [AIRCRAFT], "now": 1790400021000})
    older = parse_response({"ac": [AIRCRAFT], "now": 1790400019000})
    enrich_snapshot(newest, engine, detection_cache, timestamp_cache)
    enriched = enrich_snapshot(older, engine, detection_cache, timestamp_cache)
    assert enriched[0]["detection"]["timestamp"] == newest[0]["timestamp"]


def test_large_live_snapshot_avoids_pairwise_fleet_scan():
    class Result:
        def to_mapping(self):
            return {
                "icao24": "a00000", "flight_id": "test", "timestamp": 1,
                "risk_score": 0.0, "severity": "normal", "anomalies": [],
            }

    class PerAircraftEngine:
        updates = 0

        def update(self, state):
            self.updates += 1
            return Result()

        def update_fleet(self, states):
            raise AssertionError("large live feeds must avoid all-pairs conflict screening")

    template = parse_response({"ac": [AIRCRAFT], "now": 1790400021000})[0]
    states = [{**template, "icao24": f"a{i:05x}"} for i in range(129)]
    engine = PerAircraftEngine()
    enriched = enrich_snapshot(states, engine, {}, {})
    assert len(enriched) == 129
    assert engine.updates == 129


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
