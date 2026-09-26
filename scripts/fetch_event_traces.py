#!/usr/bin/env python3
"""Fetch and normalize ADSB.lol historical traces listed in data/events.json.

Uses only the Python standard library. Raw responses are retained under
data/raw/adsb_lol; normalized observations are written as JSON Lines under
data/processed. The script is intentionally manifest-driven and never guesses
an aircraft identity from a callsign.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "events.json"
DEFAULT_RAW = ROOT / "data" / "raw" / "adsb_lol"
DEFAULT_PROCESSED = ROOT / "data" / "processed"
BASE_URL = "https://adsb.lol/globe_history"
USER_AGENT = "Syren-Research-Prototype/0.1 (historical trace collection)"


class TraceError(Exception):
    """Raised when a response is not a valid aircraft trace."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default: 1)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--offline", action="store_true", help="normalize already-downloaded raw files only")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("events"), list):
        raise ValueError(f"Unsupported or invalid event manifest: {path}")
    for event in data["events"]:
        if not event.get("event_id") or not event.get("label_type"):
            raise ValueError("Every event must have event_id and label_type")
        for aircraft in event.get("aircraft", []):
            hex_id = aircraft.get("icao24")
            if hex_id is not None and (len(hex_id) != 6 or any(c not in "0123456789abcdefABCDEF" for c in hex_id)):
                raise ValueError(f"Invalid ICAO24 {hex_id!r} in {event['event_id']}")
    return data


def trace_url(date: str, hex_id: str) -> str:
    parsed = datetime.strptime(date, "%Y-%m-%d")
    hex_id = hex_id.lower()
    return (
        f"{BASE_URL}/{parsed:%Y/%m/%d}/traces/{hex_id[-2:]}/"
        f"trace_full_{hex_id}.json"
    )


def fetch_bytes(url: str, retries: int, delay: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/gzip, */*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise TraceError(f"HTTP {response.status} for {url}")
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TraceError(f"Trace not found (HTTP 404): {url}") from exc
            last_error = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504} or attempt == retries:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
        time.sleep(max(delay, 0.0) * (2**attempt))
    raise TraceError(f"Could not fetch {url}: {last_error}")


def decode_trace(payload: bytes) -> dict[str, Any]:
    if payload[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(payload)
        except OSError as exc:
            raise TraceError(f"Invalid gzip payload: {exc}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceError(f"Response is neither JSON nor gzip JSON: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("trace"), list):
        raise TraceError("Trace response must be a JSON object with a trace array")
    return value


def load_raw_trace(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return decode_trace(payload)


def as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def trace_points(trace: dict[str, Any], expected_hex: str) -> Iterable[dict[str, Any]]:
    actual_hex = str(trace.get("icao", trace.get("hex", ""))).lower()
    if actual_hex and actual_hex != expected_hex.lower():
        raise TraceError(f"ICAO mismatch: requested {expected_hex}, file says {actual_hex}")
    base = as_finite_number(trace.get("timestamp"))
    if base is None:
        raise TraceError("Trace is missing a numeric Unix timestamp")

    flight_leg_index = 0
    first_valid_point = True
    callsign = trace.get("flight")
    for index, point in enumerate(trace["trace"]):
        if not isinstance(point, list) or len(point) < 8:
            continue
        offset, lat, lon = (as_finite_number(point[i]) for i in range(3))
        if offset is None:
            continue
        flags = point[6] if isinstance(point[6], int) else None
        leg_start = bool(flags is not None and flags & 2)
        if not first_valid_point and leg_start:
            flight_leg_index += 1
        first_valid_point = False
        altitude_raw = point[3]
        is_ground = altitude_raw == "ground"
        altitude = as_finite_number(altitude_raw)
        timestamp = base + offset
        extra = point[8] if len(point) > 8 and isinstance(point[8], dict) else None
        if extra and extra.get("flight"):
            callsign = str(extra["flight"]).strip()
        record: dict[str, Any] = {
            "icao24": expected_hex.lower(),
            "registration": trace.get("r"),
            "aircraft_type": trace.get("t"),
            "callsign": callsign,
            "timestamp_utc": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "timestamp_unix_s": timestamp,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "altitude_baro_ft": altitude,
            "on_ground": is_ground,
            "ground_speed_kt": as_finite_number(point[4]),
            "track_deg": as_finite_number(point[5]),
            "trace_flags": point[6] if isinstance(point[6], int) else None,
            "flight_leg_index": flight_leg_index,
            "leg_start": leg_start,
            "vertical_rate_fpm": as_finite_number(point[7]),
            "source": point[9] if len(point) > 9 and isinstance(point[9], str) else None,
            "altitude_geom_ft": as_finite_number(point[10]) if len(point) > 10 else None,
            "vertical_rate_geom_fpm": as_finite_number(point[11]) if len(point) > 11 else None,
            "ias_kt": as_finite_number(point[12]) if len(point) > 12 else None,
            "roll_deg": as_finite_number(point[13]) if len(point) > 13 else None,
            "extra": extra,
            "point_index": index,
        }
        # Preserve missing coordinates as missing rather than emitting invalid rows.
        if record["latitude_deg"] is None or record["longitude_deg"] is None:
            continue
        yield record


def safe_component(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def raw_path(raw_dir: Path, event_id: str, date: str, hex_id: str) -> Path:
    return raw_dir / safe_component(event_id) / date / f"{hex_id.lower()}.json.gz"


def existing_raw_path(raw_dir: Path, event_id: str, date: str, hex_id: str) -> Path:
    compressed = raw_path(raw_dir, event_id, date, hex_id)
    uncompressed = compressed.with_suffix("")
    if compressed.exists():
        return compressed
    if uncompressed.exists():
        return uncompressed
    return compressed


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()
    if args.retries < 0 or args.delay < 0:
        print("--retries and --delay must be non-negative", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 2

    observations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    request_count = 0
    for event in manifest["events"]:
        event_id = event["event_id"]
        for aircraft in event.get("aircraft", []):
            hex_id = aircraft.get("icao24")
            if not aircraft.get("adsb_history_expected", True) or not hex_id:
                results.append({
                    "event_id": event_id,
                    "icao24": hex_id,
                    "status": "not_expected",
                    "reason": aircraft.get("collection_notes", "No public ADS-B history expected for this aircraft."),
                })
                continue
            dates = aircraft.get("trace_dates_utc", [])
            for date in dates:
                source_url = trace_url(date, hex_id)
                target = existing_raw_path(args.raw_dir, event_id, date, hex_id)
                try:
                    if not args.offline and not target.exists():
                        payload = fetch_bytes(source_url, args.retries, args.delay)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        # The endpoint serves gzip JSON under a .json name. Preserve bytes;
                        # use .json.gz for gzip bytes so tools do not guess at content.
                        if payload[:2] != b"\x1f\x8b":
                            target = target.with_suffix("")
                        target.write_bytes(payload)
                        request_count += 1
                        time.sleep(args.delay)
                    if not target.exists():
                        raise TraceError(f"No cached trace at {target}; run without --offline")
                    parsed = load_raw_trace(target)
                    expected_registration = aircraft.get("registration")
                    actual_registration = parsed.get("r")
                    if expected_registration and actual_registration:
                        if expected_registration.strip().upper() != str(actual_registration).strip().upper():
                            raise TraceError(
                                f"Registration mismatch: manifest says {expected_registration}, "
                                f"trace says {actual_registration}"
                            )
                    points = list(trace_points(parsed, hex_id))
                    if not points:
                        raise TraceError("Trace contains no valid position observations")
                    event_time = event.get("event_time_utc")
                    event_unix = datetime.fromisoformat(event_time.replace("Z", "+00:00")).timestamp() if event_time else None
                    first_unix = points[0]["timestamp_unix_s"]
                    last_unix = points[-1]["timestamp_unix_s"]
                    event_in_trace = (
                        None if event_unix is None else first_unix <= event_unix <= last_unix
                    )
                    event_leg_index = None
                    if event_unix is not None and event_in_trace:
                        for leg_index in sorted({point["flight_leg_index"] for point in points}):
                            leg_points = [point for point in points if point["flight_leg_index"] == leg_index]
                            if leg_points[0]["timestamp_unix_s"] <= event_unix <= leg_points[-1]["timestamp_unix_s"]:
                                event_leg_index = leg_index
                                break
                    for point in points:
                        observations.append({
                            "associated_event_candidate_id": event_id,
                            "event_type": event["event_type"],
                            "label_type": event["label_type"],
                            "outcome_label_at_sample": None,
                            "labeling_status": "event_time_supervision_not_generated",
                            "aircraft_role": aircraft.get("role"),
                            **point,
                        })
                    results.append({
                        "event_id": event_id,
                        "icao24": hex_id.lower(),
                        "date_utc": date,
                        "status": "downloaded_and_validated",
                        "source_url": source_url,
                        "raw_file": str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target),
                        "registration_in_trace": parsed.get("r"),
                        "aircraft_type_in_trace": parsed.get("t"),
                        "point_count": len(points),
                        "coverage_start_utc": points[0]["timestamp_utc"],
                        "coverage_end_utc": points[-1]["timestamp_utc"],
                        "event_time_covered": event_in_trace,
                        "event_flight_leg_index": event_leg_index,
                        "labeling_status": (
                            "coverage_unknown_event_time_missing" if event_unix is None
                            else "event_time_inside_trace" if event_in_trace
                            else "event_time_outside_trace"
                        ),
                        "event_time_utc": event_time,
                        "note": "A trace downloaded successfully does not establish that ADS-B contains a predictive precursor.",
                    })
                    print(f"OK {event_id} {hex_id} {date}: {len(points)} points; event covered={event_in_trace}")
                except (OSError, TraceError, ValueError, OverflowError) as exc:
                    results.append({
                        "event_id": event_id,
                        "icao24": hex_id.lower(),
                        "date_utc": date,
                        "status": "error",
                        "source_url": source_url,
                        "error": str(exc),
                    })
                    print(f"ERROR {event_id} {hex_id} {date}: {exc}", file=sys.stderr)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    obs_path = args.processed_dir / "observations.jsonl"
    count = write_jsonl(obs_path, observations)
    result_path = args.processed_dir / "fetch_results.json"
    result_path.write_text(json.dumps({
        "manifest": str(args.manifest),
        "request_count": request_count,
        "observation_count": count,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    errors = sum(result["status"] == "error" for result in results)
    print(f"Wrote {count} normalized observations to {obs_path}")
    print(f"Wrote collection report to {result_path}; {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
