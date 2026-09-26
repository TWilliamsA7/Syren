"""Quick V1 smoke, JSONL replay, and per-aircraft throughput harness.

Run ``python -m detection.harness`` for generated scenarios. All input formats
are the same mappings documented in docs/protocol.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from detection.engine import DetectionEngine, build_default_engine
from detection.models import DetectionResult, FlightState


def make_state(
    *,
    timestamp: float,
    icao24: str,
    altitude_ft: float = 10_000.0,
    vertical_rate_fpm: float | None = 0.0,
    speed_kts: float | None = 450.0,
    track_deg: float | None = 90.0,
    latitude: float = 0.0,
    longitude: float = 0.0,
    squawk: str = "1200",
    seen_age_s: float = 0.5,
    stale: bool = False,
) -> dict[str, Any]:
    """Create one protocol-shaped sample, leaving unsupported values absent."""
    return {
        "timestamp": timestamp,
        "icao24": icao24,
        "flight_id": icao24.upper(),
        "aircraft": {"registration": None, "type_code": None, "category": "A3"},
        "position": {
            "latitude": latitude,
            "longitude": longitude,
            "altitude_baro_ft": altitude_ft,
            "altitude_geom_ft": None,
            "on_ground": False,
            "source": "sim",
            "accuracy_m": 20.0,
            "stale": stale,
        },
        "kinematics": {
            "ground_speed_kts": speed_kts,
            "track_deg": track_deg,
            "vertical_rate_baro_fpm": vertical_rate_fpm,
            "vertical_rate_geom_fpm": None,
        },
        "nav": {"selected_altitude_ft": 10_000.0},
        "status": {"squawk": squawk, "emergency": "none", "seen_age_s": seen_age_s},
        "origin": "sim",
    }


def smoke_scenarios() -> tuple[tuple[str, tuple[dict[str, Any], ...], str | None], ...]:
    """Return deterministic samples and the expected anomaly type per case."""
    return (
        ("normal_baseline", tuple(make_state(timestamp=t, icao24="normal01") for t in (0, 10, 20)), None),
        ("rapid_descent", (make_state(timestamp=0, icao24="rapid01", vertical_rate_fpm=-4200),), "RAPID_DESCENT"),
        (
            "accelerating_descent",
            tuple(make_state(timestamp=t, icao24="accel01", vertical_rate_fpm=rate) for t, rate in ((0, -100), (10, -300), (20, -500))),
            "ACCELERATING_DESCENT",
        ),
        (
            "speed_decay",
            tuple(make_state(timestamp=t, icao24="speed01", speed_kts=speed) for t, speed in ((0, 450), (30, 420), (60, 390))),
            "SPEED_ANOMALY",
        ),
        (
            "abrupt_track_change",
            (make_state(timestamp=0, icao24="track01", track_deg=90), make_state(timestamp=10, icao24="track01", track_deg=150)),
            "HEADING_ANOMALY",
        ),
        (
            "heading_oscillation",
            tuple(make_state(timestamp=i * 3, icao24="oscill01", track_deg=track) for i, track in enumerate((0, 60, 0, 60, 0, 60))),
            "HEADING_OSCILLATION",
        ),
        (
            "altitude_oscillation",
            tuple(make_state(timestamp=i * 10, icao24="altosc01", altitude_ft=alt) for i, alt in enumerate((10_000, 12_000, 10_000, 12_000, 10_000))),
            "ALTITUDE_ANOMALY",
        ),
        ("emergency_squawk", (make_state(timestamp=0, icao24="squawk01", squawk="7700"),), "EMERGENCY_SQUAWK"),
        (
            "repeated_telemetry_gap",
            (make_state(timestamp=0, icao24="gap0001", seen_age_s=25, stale=True), make_state(timestamp=10, icao24="gap0001", seen_age_s=22, stale=True)),
            "TELEMETRY_GAP",
        ),
        (
            "altitude_rate_disagreement",
            (make_state(timestamp=0, icao24="mismatch", altitude_ft=10_000), make_state(timestamp=10, icao24="mismatch", altitude_ft=12_000)),
            "TELEMETRY_INCONSISTENCY",
        ),
    )


def run_smoke_suite() -> tuple[int, int]:
    """Run every generated scenario and print a compact pass/fail report."""
    passed = 0
    scenarios = smoke_scenarios()
    for name, states, expected_type in scenarios:
        engine = build_default_engine()
        try:
            results = [engine.update(state) for state in states]
            found = {anomaly.type for result in results for anomaly in result.anomalies}
            if expected_type is None:
                success = not found
                detail = "no anomalies" if success else f"unexpected {', '.join(sorted(found))}"
            else:
                success = expected_type in found
                detail = f"detected {expected_type}" if success else f"missing {expected_type}; got {', '.join(sorted(found)) or 'no anomalies'}"
            marker = "PASS" if success else "FAIL"
            print(f"{marker} {name}: {detail}")
            passed += int(success)
        except Exception as error:  # report scenario failures without hiding later cases
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    conflict_engine = build_default_engine()
    first = make_state(timestamp=100, icao24="conflict1", latitude=0, longitude=0, track_deg=90, speed_kts=360)
    second = make_state(timestamp=100, icao24="conflict2", latitude=0, longitude=10 / 60, track_deg=270, speed_kts=360)
    try:
        conflict_results = conflict_engine.update_fleet((first, second))
        conflict_found = all(
            any(anomaly.type == "AIRCRAFT_CONFLICT" for anomaly in result.anomalies)
            for result in conflict_results
        )
        print(f"{'PASS' if conflict_found else 'FAIL'} aircraft_conflict: {'detected for both aircraft' if conflict_found else 'expected conflict alert for both aircraft'}")
        passed += int(conflict_found)
    except Exception as error:
        print(f"FAIL aircraft_conflict: {type(error).__name__}: {error}")
    total = len(scenarios) + 1
    print(f"Smoke result: {passed}/{total} scenarios passed")
    return passed, total


def replay_jsonl(path: Path, *, fleet_snapshots: bool = False) -> int:
    """Replay protocol JSONL; fleet mode expects one JSON array per line."""
    engine = build_default_engine()
    line_count = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if fleet_snapshots:
                    if not isinstance(record, list):
                        raise ValueError("fleet snapshot line must be a JSON array")
                    results = engine.update_fleet(record)
                    outputs = [result.to_mapping() for result in results]
                else:
                    if not isinstance(record, Mapping):
                        raise ValueError("record line must be a JSON object")
                    outputs = [engine.update(record).to_mapping()]
                for output in outputs:
                    print(json.dumps(output, separators=(",", ":")))
                    line_count += 1
            except Exception as error:
                print(f"{path}:{line_number}: {type(error).__name__}: {error}", file=sys.stderr)
                return 1
    print(f"Replayed {line_count} state result(s)", file=sys.stderr)
    return 0


def benchmark_aircraft(counts: Iterable[int]) -> int:
    """Measure per-aircraft detector throughput, excluding pairwise conflict scan."""
    for count in counts:
        if count <= 0:
            print("benchmark sizes must be positive", file=sys.stderr)
            return 2
        engine = build_default_engine()
        start = time.perf_counter()
        for index in range(count):
            engine.update(make_state(timestamp=0, icao24=f"{index:06x}"[-6:]))
        elapsed = time.perf_counter() - start
        rate = count / elapsed if elapsed else float("inf")
        print(f"{count} aircraft states: {elapsed:.3f}s, {rate:.1f} states/s (single-state path; conflict scan excluded)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--input-jsonl", type=Path, help="replay one protocol FlightState JSON object per line")
    mode.add_argument("--fleet-jsonl", type=Path, help="replay one JSON array of synchronized FlightStates per line")
    mode.add_argument("--benchmark", nargs="+", type=int, metavar="AIRCRAFT_COUNT", help="time single-state processing at these fleet sizes")
    args = parser.parse_args(argv)
    if args.input_jsonl:
        return replay_jsonl(args.input_jsonl)
    if args.fleet_jsonl:
        return replay_jsonl(args.fleet_jsonl, fleet_snapshots=True)
    if args.benchmark:
        return benchmark_aircraft(args.benchmark)
    passed, total = run_smoke_suite()
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
