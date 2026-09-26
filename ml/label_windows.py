#!/usr/bin/env python3
"""Build causal pre-declaration windows from retained ADSB.lol aircraft traces.

This first training target is an observed ADS-B emergency declaration, not an
independently confirmed accident or operational emergency. No network access or
third-party package is needed.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ml.features import FEATURE_NAMES, extract_features
from scripts.collect_adsb_days import inspect_trace
from scripts.fetch_event_traces import TraceError, load_raw_trace, trace_points

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = ROOT / "data" / "collection"
DEFAULT_OUTPUT = ROOT / "data" / "learning"
TARGET = "first_observed_adsb_emergency_declaration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-seconds", type=float, default=300)
    parser.add_argument("--min-lead-seconds", type=float, default=120)
    parser.add_argument("--horizon-seconds", type=float, default=600)
    parser.add_argument("--stride-seconds", type=float, default=60)
    parser.add_argument("--max-gap-seconds", type=float, default=120)
    parser.add_argument("--min-points", type=int, default=10)
    return parser.parse_args()


def utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_jsonl_files(directory: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected an object at {path}:{line_number}")
                    yield value


def resolve_trace(collection_dir: Path, relative: str) -> Path:
    base = collection_dir.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Trace path escapes collection directory: {relative}")
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def segments(points: list[dict[str, Any]], max_gap_s: float) -> Iterator[list[dict[str, Any]]]:
    segment: list[dict[str, Any]] = []
    for point in points:
        current = point["timestamp_unix_s"]
        if segment:
            previous = segment[-1]
            new_segment = (
                point["flight_leg_index"] != previous["flight_leg_index"]
                or current - previous["timestamp_unix_s"] > max_gap_s
            )
            if current <= previous["timestamp_unix_s"]:
                continue
            if new_segment:
                yield segment
                segment = []
        segment.append(point)
    if segment:
        yield segment


def checked_trace(record: dict[str, Any], collection_dir: Path, positive_manifest: bool) -> tuple[list[dict[str, Any]], float | None]:
    relative = record.get("trace_file")
    icao24 = str(record.get("icao24", "")).lower()
    if not isinstance(relative, str) or len(icao24) != 6:
        raise ValueError(f"Missing trace_file or ICAO24 in manifest: {record}")
    source = resolve_trace(collection_dir, relative)
    trace = load_raw_trace(source)
    actual = str(trace.get("icao", trace.get("hex", ""))).lower()
    if actual != icao24:
        raise ValueError(f"ICAO24 mismatch in {source}: manifest={icao24}, trace={actual}")
    first = datetime.fromtimestamp(float(trace["timestamp"]), tz=timezone.utc).date().isoformat()
    if record.get("date_utc") != first:
        raise ValueError(f"UTC date mismatch in {source}: manifest={record.get('date_utc')}, trace={first}")
    signal_info = inspect_trace(source.read_bytes())
    if signal_info is None:
        raise ValueError(f"Invalid trace payload: {source}")
    actual_signals = signal_info["signals"]
    if positive_manifest and not actual_signals:
        raise ValueError(f"Candidate manifest has no signal in raw trace: {source}")
    if not positive_manifest and actual_signals:
        raise ValueError(f"Control manifest contains an emergency signal: {source}")
    event_time: float | None = None
    if positive_manifest:
        expected = record.get("signals")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"Candidate has no signal evidence: {relative}")
        actual_offset = min(float(item["offset_s"]) for item in actual_signals)
        expected_offset = min(float(item["offset_s"]) for item in expected)
        if abs(actual_offset - expected_offset) > 0.5:
            raise ValueError(f"Signal onset mismatch in {source}: manifest={expected_offset}, trace={actual_offset}")
        event_time = float(trace["timestamp"]) + actual_offset
    try:
        points = list(trace_points(trace, icao24))
    except (TraceError, OverflowError) as exc:
        raise ValueError(f"Invalid position trace {source}: {exc}") from exc
    return points, event_time


def labeled_rows(
    points: list[dict[str, Any]],
    event_time: float | None,
    record: dict[str, Any],
    config: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    for segment_index, segment in enumerate(segments(points, config.max_gap_seconds)):
        if len(segment) < config.min_points:
            continue
        times = [point["timestamp_unix_s"] for point in segment]
        last_anchor = -math.inf
        for index, anchor in enumerate(times):
            if anchor - last_anchor < config.stride_seconds:
                continue
            if anchor - times[0] < config.context_seconds:
                continue
            start = anchor - config.context_seconds
            left = bisect.bisect_left(times, start)
            history = segment[left:index + 1]
            if len(history) < config.min_points or history[0]["timestamp_unix_s"] > start + config.max_gap_seconds:
                continue
            if event_time is not None and anchor >= event_time - config.min_lead_seconds:
                continue
            positive = (
                event_time is not None
                and times[0] <= event_time <= times[-1]
                and config.min_lead_seconds < event_time - anchor <= config.horizon_seconds
            )
            if not positive and times[-1] < anchor + config.horizon_seconds:
                continue
            features = extract_features(history)
            if tuple(features) != FEATURE_NAMES:
                raise ValueError("Feature schema changed unexpectedly")
            last_anchor = anchor
            day = record["date_utc"]
            icao24 = record["icao24"].lower()
            leg_index = segment[0]["flight_leg_index"]
            group_id = f"{day}:{icao24}:{leg_index}:{segment_index}"
            digest = hashlib.sha256(f"{group_id}:{anchor:.3f}".encode()).hexdigest()[:20]
            yield {
                "schema_version": 1,
                "example_id": digest,
                "target": TARGET,
                "label": int(positive),
                "date_utc": day,
                "icao24": icao24,
                "group_id": group_id,
                "flight_leg_index": leg_index,
                "anchor_unix_s": anchor,
                "anchor_utc": utc(anchor),
                "event_unix_s": event_time if positive else None,
                "event_utc": utc(event_time) if positive and event_time is not None else None,
                "trace_file": record["trace_file"],
                "features": features,
            }


def build_dataset(config: argparse.Namespace) -> dict[str, Any]:
    if not 0 < config.min_lead_seconds < config.horizon_seconds:
        raise ValueError("Require 0 < min lead < horizon")
    if min(config.context_seconds, config.stride_seconds, config.max_gap_seconds) <= 0 or config.min_points < 2:
        raise ValueError("Context, stride, gap and min-points must be positive")
    if not (config.collection_dir / "reports").is_dir():
        raise ValueError(f"Missing completed-day reports in {config.collection_dir}")
    records = [
        (True, record) for record in load_jsonl_files(config.collection_dir / "emergency_candidates")
    ] + [
        (False, record) for record in load_jsonl_files(config.collection_dir / "control_manifests")
    ]
    if not records or not any(positive for positive, _ in records):
        raise ValueError("No candidate and control manifests found")
    completed_dates = {path.stem for path in (config.collection_dir / "reports").glob("*.json")}
    keys: set[tuple[str, str]] = set()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output = config.output_dir / "declaration_windows.jsonl"
    partial = output.with_suffix(".jsonl.tmp")
    counts: Counter[str] = Counter()
    per_date: dict[str, Counter[str]] = {}
    candidates_without_positive: list[str] = []
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            for positive_manifest, record in records:
                day = str(record.get("date_utc"))
                icao24 = str(record.get("icao24", "")).lower()
                key = day, icao24
                if day not in completed_dates:
                    raise ValueError(f"Missing completed-day report for {day}")
                if key in keys:
                    raise ValueError(f"Duplicate aircraft-day across manifests: {key}")
                keys.add(key)
                points, event_time = checked_trace(record, config.collection_dir, positive_manifest)
                counts["candidate_traces" if positive_manifest else "control_traces"] += 1
                trace_positive = 0
                for example in labeled_rows(points, event_time, record, config):
                    stream.write(json.dumps(example, separators=(",", ":"), allow_nan=False) + "\n")
                    label_name = "positive_windows" if example["label"] else "negative_windows"
                    counts[label_name] += 1
                    per_date.setdefault(day, Counter())[label_name] += 1
                    trace_positive += example["label"]
                if positive_manifest and trace_positive == 0:
                    candidates_without_positive.append(f"{day}:{icao24}")
        for attempt in range(15):
            try:
                partial.replace(output)
                break
            except PermissionError:
                if attempt == 14:
                    raise
                time.sleep(0.4)
    finally:
        if partial.exists():
            for attempt in range(15):
                try:
                    partial.unlink()
                    break
                except PermissionError:
                    if attempt == 14:
                        raise
                    time.sleep(0.4)
    summary = {
        "schema_version": 1,
        "target": TARGET,
        "target_meaning": "First observed ADS-B emergency status or squawk 7500/7600/7700 within the lead-time window; this is not independently confirmed incident risk.",
        "collection_dir": str(config.collection_dir.resolve()),
        "examples_file": str(output.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "config": {
            "context_seconds": config.context_seconds,
            "min_lead_seconds": config.min_lead_seconds,
            "horizon_seconds": config.horizon_seconds,
            "stride_seconds": config.stride_seconds,
            "max_gap_seconds": config.max_gap_seconds,
            "min_points": config.min_points,
        },
        "counts": dict(counts),
        "per_date": {day: dict(date_counts) for day, date_counts in sorted(per_date.items())},
        "candidates_without_positive_windows": candidates_without_positive,
        "warning": "Dates and aircraft are heavily sampled; negative windows mean no declaration observed during a covered horizon, not no real-world emergency. Prevalence and risk calibration are unavailable from this cohort.",
    }
    (config.output_dir / "declaration_windows_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = build_dataset(args)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"Labeling failed: {exc}")
        return 1
    print(json.dumps({"target": TARGET, "counts": summary["counts"], "per_date": summary["per_date"]}, indent=2))
    print(f"Wrote {summary['examples_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
