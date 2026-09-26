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

from ml.features import FEATURE_NAMES, TEMPORAL_FEATURE_NAMES, extract_features, extract_temporal_features
from ml.sequence_features import CHANNEL_NAMES, resample_sequence
from scripts.collect_adsb_days import inspect_trace
from scripts.fetch_event_traces import TraceError, load_raw_trace, trace_points

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = ROOT / "data" / "collection"
DEFAULT_OUTPUT = ROOT / "data" / "learning"
TARGET = "first_observed_adsb_emergency_declaration"
EPISODE_TARGET = "future_adsb_declaration_episode_multilabel_2_to_10_minutes"
SUPPORTED_SUBTYPE_HEADS = (
    "emergency_field:general",
    "emergency_field:nordo",
    "emergency_squawk:7700",
    "emergency_squawk:7600",
)
USABLE_STATUSES = {"general", "minfuel", "nordo", "unlawful", "downed"}
USABLE_SQUAWKS = {"7700", "7600", "7500"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-file", type=Path, default=DEFAULT_OUTPUT / "candidate_audit.jsonl")
    parser.add_argument("--context-seconds", type=float, default=300)
    parser.add_argument("--min-lead-seconds", type=float, default=120)
    parser.add_argument("--horizon-seconds", type=float, default=600)
    parser.add_argument("--stride-seconds", type=float, default=60)
    parser.add_argument("--max-gap-seconds", type=float, default=120)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument(
        "--episode-target", action="store_true",
        help="write the separate time-clustered, subtype-multilabel research dataset",
    )
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


def airborne_exposure(segment: list[dict[str, Any]], index: int, seconds: float) -> float:
    """Observed airborne time after an anchor, capped at the next stride."""
    anchor = segment[index]["timestamp_unix_s"]
    end = anchor + seconds
    total = 0.0
    for j in range(index, len(segment) - 1):
        left, right = segment[j], segment[j + 1]
        if left["timestamp_unix_s"] >= end:
            break
        if left.get("on_ground") is False and right.get("on_ground") is False:
            total += max(0.0, min(end, right["timestamp_unix_s"]) - left["timestamp_unix_s"])
    return total


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


def usable_subtype(signal: dict[str, Any]) -> str | None:
    kind = signal.get("signal")
    value = str(signal.get("value", "")).lower()
    if kind == "emergency_field" and value in USABLE_STATUSES:
        return f"emergency_field:{value}"
    if kind == "emergency_squawk" and value in USABLE_SQUAWKS:
        return f"emergency_squawk:{value}"
    return None


def build_signal_episodes(
    usable_onset_unix_s: float, signals: list[dict[str, Any]], date_utc: str, icao24: str,
    cluster_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Cluster usable signal transitions into bounded 60-second multi-label episodes."""
    usable = []
    for signal in signals:
        subtype = usable_subtype(signal)
        offset = signal.get("offset_s")
        if subtype is None or isinstance(offset, bool) or not isinstance(offset, (int, float)):
            continue
        usable.append((float(offset), subtype))
    usable.sort()
    if not usable:
        return []
    offset_origin = usable[0][0]
    episodes: list[dict[str, Any]] = []
    for offset, subtype in usable:
        event_time = float(usable_onset_unix_s) + offset - offset_origin
        if not episodes or event_time - episodes[-1]["event_unix_s"] > cluster_seconds:
            episodes.append({"event_unix_s": event_time, "signal_subtypes": set()})
        episodes[-1]["signal_subtypes"].add(subtype)
    for episode in episodes:
        episode["signal_subtypes"] = sorted(episode["signal_subtypes"])
        material = f"{date_utc}:{icao24.lower()}:{episode['event_unix_s']:.3f}"
        episode["event_id"] = hashlib.sha256(material.encode()).hexdigest()[:20]
    return episodes


def labeled_rows(
    points: list[dict[str, Any]],
    event_time: float | None,
    record: dict[str, Any],
    config: argparse.Namespace,
    event_signal_subtypes: list[str] | None = None,
    event_episodes: list[dict[str, Any]] | None = None,
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
            future_events: list[dict[str, Any]] = []
            near_event = False
            if event_episodes is not None:
                segment_events = [
                    event for event in event_episodes
                    if times[0] <= float(event["event_unix_s"]) <= times[-1]
                ]
                deltas = [float(event["event_unix_s"]) - anchor for event in segment_events]
                near_event = any(0 < delta <= config.min_lead_seconds for delta in deltas)
                future_events = [
                    event for event, delta in zip(segment_events, deltas)
                    if config.min_lead_seconds < delta <= config.horizon_seconds
                ]
                if near_event:
                    continue
                positive = bool(future_events)
            else:
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
            temporal_features = extract_temporal_features(history)
            sequence = resample_sequence(history, anchor)
            if tuple(features) != FEATURE_NAMES:
                raise ValueError("Feature schema changed unexpectedly")
            last_anchor = anchor
            day = record["date_utc"]
            icao24 = record["icao24"].lower()
            leg_index = segment[0]["flight_leg_index"]
            group_id = f"{day}:{icao24}:{leg_index}:{segment_index}"
            digest = hashlib.sha256(f"{group_id}:{anchor:.3f}".encode()).hexdigest()[:20]
            episode_mode = event_episodes is not None
            event_time_for_row = (
                float(future_events[0]["event_unix_s"]) if episode_mode and future_events
                else event_time if positive else None
            )
            subtype_names = sorted({
                subtype for event in future_events for subtype in event["signal_subtypes"]
            }) if episode_mode else list(event_signal_subtypes or ()) if positive else []
            subtype_targets = {name: int(name in subtype_names) for name in subtype_names}
            targets = {"any_declaration": int(positive)}
            targets.update({name: int(name in subtype_names) for name in SUPPORTED_SUBTYPE_HEADS})
            yield {
                "schema_version": 3 if episode_mode else 2,
                "example_id": digest,
                "target": EPISODE_TARGET if episode_mode else TARGET,
                "label": int(positive),
                "targets": targets if episode_mode else None,
                "subtype_targets": subtype_targets if episode_mode else None,
                "date_utc": day,
                "icao24": icao24,
                "group_id": group_id,
                "flight_leg_index": leg_index,
                "anchor_unix_s": anchor,
                "anchor_utc": utc(anchor),
                "event_unix_s": event_time_for_row,
                "event_utc": utc(event_time_for_row) if event_time_for_row is not None else None,
                "future_events": [
                    {"event_id": event["event_id"], "event_unix_s": event["event_unix_s"],
                     "signal_subtypes": list(event["signal_subtypes"])}
                    for event in future_events
                ] if episode_mode else None,
                "trace_file": record["trace_file"],
                "source_cohort": "candidate" if event_time is not None else "control_sample",
                "features": features,
                "temporal_features": temporal_features,
                "sequence_schema_version": 2 if episode_mode else 1,
                "sequence": sequence,
                # Evaluation metadata only. This field is never passed to a model.
                "event_signal_subtypes": subtype_names if positive else [],
                "airborne_exposure_seconds": airborne_exposure(segment, index, config.stride_seconds),
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
    if not config.audit_file.is_file():
        raise ValueError(f"Missing candidate audit; run python -m ml.audit_candidates first: {config.audit_file}")
    audit = {}
    with config.audit_file.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item["date_utc"], item["icao24"])
            if key in audit:
                raise ValueError(f"Duplicate audit row: {key}")
            audit[key] = item
    candidate_keys = {(record["date_utc"], record["icao24"].lower())
                      for positive, record in records if positive}
    if set(audit) != candidate_keys:
        raise ValueError(f"Candidate audit does not match manifests; missing={candidate_keys - set(audit)}, extra={set(audit) - candidate_keys}")
    completed_dates = {path.stem for path in (config.collection_dir / "reports").glob("*.json")}
    keys: set[tuple[str, str]] = set()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    episode_mode = bool(getattr(config, "episode_target", False))
    output_stem = "declaration_episode_windows" if episode_mode else "declaration_windows"
    output = config.output_dir / f"{output_stem}.jsonl"
    partial = output.with_suffix(".jsonl.tmp")
    counts: Counter[str] = Counter()
    per_date: dict[str, Counter[str]] = {}
    candidates_without_positive: list[str] = []
    episode_count = 0
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
                if positive_manifest:
                    decision = audit.get(key)
                    if decision is None or decision["trace_file"] != record["trace_file"]:
                        raise ValueError(f"Missing or mismatched audit decision: {key}")
                    if decision["decision"] == "exclude_proxy":
                        counts["excluded_candidate_traces"] += 1
                        continue
                    if decision["decision"] != "include_proxy":
                        raise ValueError(f"Unknown audit decision: {key}")
                points, event_time = checked_trace(record, config.collection_dir, positive_manifest)
                event_signal_subtypes: list[str] = []
                event_episodes: list[dict[str, Any]] | None = [] if episode_mode else None
                if positive_manifest:
                    event_time = decision["usable_onset_unix_s"]
                    if event_time is None:
                        raise ValueError(f"Included candidate has no onset: {key}")
                    if episode_mode:
                        event_episodes = build_signal_episodes(
                            event_time, decision.get("signals", []), day, icao24,
                        )
                        episode_count += len(event_episodes)
                    else:
                        event_signal_subtypes = sorted({
                            subtype for signal in decision.get("signals", [])
                            if (subtype := usable_subtype(signal)) is not None
                        })
                counts["candidate_traces" if positive_manifest else "control_traces"] += 1
                trace_positive = 0
                for example in labeled_rows(
                    points, event_time, record, config, event_signal_subtypes,
                    event_episodes=event_episodes,
                ):
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
        "schema_version": 3 if episode_mode else 2,
        "target": EPISODE_TARGET if episode_mode else TARGET,
        "target_meaning": (
            "Each transition among usable ADS-B emergency statuses or squawks is a timestamped proxy episode; transitions up to 60 seconds after the first signal are grouped into one multi-label episode. This is not independently confirmed incident risk."
            if episode_mode else
            "First audited usable ADS-B emergency status or squawk 7500/7600/7700 within the lead-time window; lifeguard-only status is excluded. This is not independently confirmed incident risk."
        ),
        "collection_dir": str(config.collection_dir.resolve()),
        "examples_file": str(output.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES),
        "sequence_schema_version": 2 if episode_mode else 1,
        "sequence_channels": list(CHANNEL_NAMES) if episode_mode else [
            "altitude_baro_ft", "ground_speed_kt", "vertical_rate_fpm",
            "track_sin", "track_cos", "on_ground", "altitude_valid",
            "ground_speed_valid", "vertical_rate_valid", "track_valid", "on_ground_valid",
        ],
        "supported_subtype_heads": list(SUPPORTED_SUBTYPE_HEADS) if episode_mode else [],
        "audit_file": str(config.audit_file.resolve()),
        "config": {
            "context_seconds": config.context_seconds,
            "min_lead_seconds": config.min_lead_seconds,
            "horizon_seconds": config.horizon_seconds,
            "stride_seconds": config.stride_seconds,
            "max_gap_seconds": config.max_gap_seconds,
            "min_points": config.min_points,
            "episode_cluster_seconds": 60 if episode_mode else None,
        },
        "counts": dict(counts),
        "event_episodes": episode_count if episode_mode else None,
        "per_date": {day: dict(date_counts) for day, date_counts in sorted(per_date.items())},
        "candidates_without_positive_windows": candidates_without_positive,
        "warning": "Dates and aircraft are heavily sampled; negative windows mean no declaration observed during a covered horizon, not no real-world emergency. Prevalence and risk calibration are unavailable from this cohort.",
    }
    (config.output_dir / f"{output_stem}_summary.json").write_text(
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
    print(json.dumps({"target": summary["target"], "counts": summary["counts"],
                      "event_episodes": summary["event_episodes"], "per_date": summary["per_date"]}, indent=2))
    print(f"Wrote {summary['examples_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
