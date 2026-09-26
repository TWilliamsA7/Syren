"""Create source-backed incident windows without mixing them with declaration labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from ml.label_windows import DEFAULT_COLLECTION, DEFAULT_OUTPUT, labeled_rows, load_jsonl_files, resolve_trace
from scripts.fetch_event_traces import load_raw_trace, raw_path, trace_points

ROOT = Path(__file__).resolve().parents[1]
TARGET = "verified_emergency_outcome_2_to_10_minutes"
PRECISION_RADIUS_SECONDS = {"second": 0, "minute": 30, "minute_estimate": 60}


def event_seconds(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"Event time must be UTC with a Z or +00:00 offset: {value}")
    return parsed.timestamp()


def robust_label(anchor: float, midpoint: float, radius: float) -> int | None:
    """Return a label only if every plausible event time gives the same class."""
    nearest_lead = midpoint - radius - anchor
    latest_lead = midpoint + radius - anchor
    if 120 < nearest_lead and latest_lead <= 600:
        return 1
    if nearest_lead > 600:
        return 0
    return None


def label(events_path: Path, raw_dir: Path, collection_dir: Path, output_dir: Path) -> dict:
    registry = json.loads(events_path.read_text(encoding="utf-8"))["events"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config = argparse.Namespace(context_seconds=300, min_lead_seconds=120, horizon_seconds=600,
                                stride_seconds=60, max_gap_seconds=120, min_points=10)
    event_rows = []
    cases = []
    incident_types = set()
    usable_phases = set()
    for event in registry:
        if event.get("label_type") not in {"accident_outcome", "incident_outcome", "verified_emergency_outcome"}:
            continue
        if not event.get("sources") or not event.get("event_time_utc") or not event.get("event_type"):
            raise ValueError(f"Event lacks source, time, or type: {event.get('event_id')}")
        precision = event.get("event_time_precision")
        if precision not in PRECISION_RADIUS_SECONDS:
            raise ValueError(f"Unsupported time precision in {event['event_id']}: {precision}")
        midpoint = event_seconds(event["event_time_utc"])
        radius = PRECISION_RADIUS_SECONDS[precision]
        for aircraft in event.get("aircraft", []):
            icao = aircraft.get("icao24")
            if not icao:
                continue
            dates = aircraft.get("trace_dates_utc", [])
            for day in dates:
                path = raw_path(raw_dir, event["event_id"], day, icao)
                case = {"event_id": event["event_id"], "event_type": event["event_type"],
                        "icao24": icao.lower(), "date_utc": day, "trace_file": str(path),
                        "source_urls": event["sources"], "identity_source": aircraft.get("identity_source"),
                        "time_precision": precision, "positive_windows": 0}
                if not path.is_file():
                    case["status"] = "trace_missing"
                    cases.append(case)
                    continue
                raw = load_raw_trace(path)
                if str(raw.get("icao", raw.get("hex", ""))).lower() != icao.lower():
                    raise ValueError(f"ICAO mismatch for {path}")
                if aircraft.get("registration") and raw.get("r") and raw["r"] != aircraft["registration"]:
                    raise ValueError(f"Registration mismatch for {path}")
                points = list(trace_points(raw, icao))
                if not points or not points[0]["timestamp_unix_s"] <= midpoint <= points[-1]["timestamp_unix_s"]:
                    case["status"] = "event_not_covered_by_trace"
                    cases.append(case)
                    continue
                record = {"date_utc": day, "icao24": icao.lower(), "trace_file": str(path)}
                for row in labeled_rows(points, midpoint, record, config):
                    stable = robust_label(row["anchor_unix_s"], midpoint, radius)
                    if stable is None or stable != row["label"]:
                        continue
                    row.update({
                        "target": TARGET, "event_id": event["event_id"],
                        "event_type": event["event_type"], "event_sources": event["sources"],
                        "time_uncertainty_seconds": radius, "label_strength": "source_verified_outcome",
                        "source_cohort": "verified_event",
                    })
                    event_rows.append(row)
                    if row["label"]:
                        case["positive_windows"] += 1
                        features = row["temporal_features"]
                        phase = (features["airborne_at_anchor"], features["climbing_at_anchor"],
                                 features["descending_at_anchor"])
                        usable_phases.add(phase)
                case["status"] = "eligible" if case["positive_windows"] else "no_eligible_positive_window"
                if case["positive_windows"]:
                    incident_types.add(str(raw.get("t")))
                cases.append(case)
    controls = []
    control_types = Counter()
    for record in load_jsonl_files(collection_dir / "control_manifests"):
        aircraft_type = str(record.get("aircraft_type"))
        if aircraft_type not in incident_types:
            continue
        path = resolve_trace(collection_dir, record["trace_file"])
        raw = load_raw_trace(path)
        points = list(trace_points(raw, record["icao24"]))
        selected = 0
        for row in labeled_rows(points, None, record, config):
            features = row["temporal_features"]
            phase = (features["airborne_at_anchor"], features["climbing_at_anchor"],
                     features["descending_at_anchor"])
            if phase not in usable_phases:
                continue
            row.update({
                "target": TARGET, "event_id": None, "event_type": None, "event_sources": [],
                "time_uncertainty_seconds": None, "label_strength": "no_registry_match_weak_negative",
                "control_match": {"aircraft_type": aircraft_type, "flight_phase": phase},
                "control_review_status": "not_independently_cleared",
            })
            controls.append(row)
            selected += 1
        if selected:
            control_types[aircraft_type] += 1
    output = output_dir / "verified_incident_windows.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in (*event_rows, *controls):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    summary = {
        "schema_version": 1, "target": TARGET, "event_cases": cases,
        "positive_windows": sum(row["label"] for row in event_rows),
        "positive_event_aircraft": sum(case["positive_windows"] > 0 for case in cases),
        "event_trace_negative_windows": sum(not row["label"] for row in event_rows),
        "weak_control_windows": len(controls), "matched_control_aircraft_by_type": dict(control_types),
        "warning": "Only two incident aircraft currently have eligible positives. Controls are weak negatives; do not train or claim incident performance yet.",
    }
    (output_dir / "verified_incident_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data" / "events.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "adsb_lol")
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(label(args.events, args.raw_dir, args.collection_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
