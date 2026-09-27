import gzip
import json
import os
import tempfile

from data.trace import export_day, open_jsonl, parse_trace
from shared.flight_state import validate_flight_state

MIDNIGHT = 1790208000.0  # 2026-09-24 00:00 UTC
FOUR_PM = MIDNIGHT + 16 * 3600


def write_trace(folder, icao, points, **extra):
    path = os.path.join(folder, "traces", icao[-2:], f"trace_full_{icao}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({"icao": icao, "timestamp": MIDNIGHT, "trace": points, **extra}, f)
    return path


def point(seconds, lat=28.4, lon=-81.3, altitude=5000, flags=0, rate=0, details=None,
          source="adsb_icao"):
    return [seconds, lat, lon, altitude, 250.0, 90.0, flags, rate, details, source,
            None, None, None, None]


def test_parse_trace():
    details = {"flight": "DAL123  ", "squawk": "3456", "category": "A3", "rc": 186,
               "nav_altitude_mcp": 7008, "emergency": "none"}
    with tempfile.TemporaryDirectory() as root:
        path = write_trace(root, "a80595", [
            point(60, details=details),
            point(65, altitude="ground"),
            point(70, altitude=5200, rate=640, flags=8 | 4),  # GPS altitude and GPS rate
            point(75, flags=1),                                # stale
            point(80, flags=2, source=None, details={"type": "mlat"}),  # a new leg
        ], r="N123DL", t="A321")
        states = parse_trace(path)
    assert [validate_flight_state(state) for state in states] == [[]] * 5
    first, ground, geom, stale, new_leg = states

    assert first["timestamp"] == MIDNIGHT + 60 and first["origin"] == "history"
    assert first["flight_id"] == "DAL123" and first["aircraft"]["registration"] == "N123DL"
    assert first["position"]["accuracy_m"] == 186 and first["nav"]["selected_altitude_ft"] == 7008
    assert ground["position"]["on_ground"] and ground["position"]["altitude_baro_ft"] is None
    assert ground["status"]["squawk"] == "3456", "details are reused until new ones come"
    assert geom["position"]["altitude_baro_ft"] is None and geom["position"]["altitude_geom_ft"] == 5200
    assert geom["kinematics"]["vertical_rate_baro_fpm"] is None
    assert geom["kinematics"]["vertical_rate_geom_fpm"] == 640
    assert stale["position"]["stale"] and not first["position"]["stale"]
    assert new_leg["flight_id"] == "a80595", "a new leg forgets the last callsign"
    assert new_leg["status"]["squawk"] is None and new_leg["position"]["source"] == "mlat"


def test_parse_trace_keeps_details_from_skipped_points():
    with tempfile.TemporaryDirectory() as root:
        path = write_trace(root, "a80595", [
            point(60, details={"flight": "DAL123", "squawk": "3456"}),
            point(65),
        ])
        states = parse_trace(path, keep=lambda timestamp, lat, lon: timestamp >= MIDNIGHT + 65)
    assert [state["timestamp"] for state in states] == [MIDNIGHT + 65]
    assert states[0]["flight_id"] == "DAL123" and states[0]["status"]["squawk"] == "3456"


def test_export_day():
    with tempfile.TemporaryDirectory() as root:
        day = os.path.join(root, "fixed", "2026-09-24")
        write_trace(day, "a80595", [
            point(16 * 3600 - 10),     # before the window
            point(16 * 3600 + 10),
            point(16 * 3600 + 1800),
            point(17 * 3600),          # the window ends at 17:00
        ])
        write_trace(day, "abc10c", [point(16 * 3600 + 5)])
        write_trace(day, "b00001", [point(16 * 3600 + 5, lat=51.5, lon=-0.1)])  # London, outside the US

        out_path = os.path.join(root, "exports", "day.jsonl.gz")
        progress = []
        counts = export_day("2026-09-24", out_path, start="16:00", hours=1, root=root,
                            on_progress=lambda done, total: progress.append((done, total)))
        assert progress[-1] == (3, 3), "the last progress report is all 3 trace files"
        with open_jsonl(out_path) as f:
            states = [json.loads(line) for line in f]

        assert counts == (3, 2)
        assert progress[0] == (0, 3) and progress[-1] == (3, 3)
        assert [done for done, _ in progress] == sorted(done for done, _ in progress)
        assert os.listdir(os.path.dirname(out_path)) == ["day.jsonl.gz"], "no partial file left"
        assert [(state["icao24"], state["timestamp"]) for state in states] == [
            ("abc10c", FOUR_PM + 5), ("a80595", FOUR_PM + 10), ("a80595", FOUR_PM + 1800)]

        try:
            export_day("2026-09-20", out_path, root=root)
            assert False, "a day that isn't on disk should be refused"
        except LookupError:
            pass


if __name__ == "__main__":
    test_parse_trace()
    test_parse_trace_keeps_details_from_skipped_points()
    test_export_day()
    print("all tests passed")
