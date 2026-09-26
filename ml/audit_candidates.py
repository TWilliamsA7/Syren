"""Audit retained declaration candidates against raw traces before labeling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ml.label_windows import DEFAULT_COLLECTION, DEFAULT_OUTPUT, checked_trace, load_jsonl_files, resolve_trace, segments, utc
from scripts.collect_adsb_days import inspect_trace
from scripts.fetch_event_traces import load_raw_trace

USABLE_STATUSES = {"general", "minfuel", "nordo", "unlawful", "downed"}
USABLE_SQUAWKS = {"7700", "7600", "7500"}


def usable_signal(signal: dict) -> bool:
    value = str(signal["value"]).lower()
    return (signal["signal"] == "emergency_field" and value in USABLE_STATUSES) or (
        signal["signal"] == "emergency_squawk" and value in USABLE_SQUAWKS)


def audit(collection_dir: Path, output_dir: Path, reviews_path: Path | None = None) -> dict:
    reviews = {}
    if reviews_path is not None and not reviews_path.is_file():
        raise FileNotFoundError(reviews_path)
    if reviews_path is not None:
        with reviews_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                review = json.loads(line)
                key = (review["date_utc"], review["icao24"].lower())
                if key in reviews or review["decision"] not in {"include_proxy", "exclude_proxy"}:
                    raise ValueError(f"Duplicate or invalid review: {key}")
                if not review.get("reviewer") or not review.get("reason"):
                    raise ValueError(f"Review requires reviewer and reason: {key}")
                reviews[key] = review
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "candidate_audit.jsonl"
    counts: Counter[str] = Counter()
    signal_values: Counter[str] = Counter()
    seen = set()
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in load_jsonl_files(collection_dir / "emergency_candidates"):
            key = (record["date_utc"], record["icao24"].lower())
            if key in seen:
                raise ValueError(f"Duplicate candidate: {key}")
            seen.add(key)
            points, _ = checked_trace(record, collection_dir, True)
            source = resolve_trace(collection_dir, record["trace_file"])
            raw = load_raw_trace(source)
            inspected = inspect_trace(source.read_bytes())
            if inspected is None:
                raise ValueError(f"Cannot inspect {source}")
            signals = sorted(inspected["signals"], key=lambda s: float(s["offset_s"]))
            for signal in signals:
                signal_values[f"{signal['signal']}:{signal['value']}"] += 1
            usable = [s for s in signals if usable_signal(s)]
            decision = "include_proxy" if usable else "exclude_proxy"
            reason = "usable_declaration_signal" if usable else "priority_status_only"
            review = reviews.get(key)
            if review is not None:
                decision, reason = review["decision"], review["reason"]
            if decision == "include_proxy" and not usable:
                raise ValueError(f"Cannot include candidate without a usable declaration signal: {key}")
            onset = float(raw["timestamp"]) + float(usable[0]["offset_s"]) if usable else None
            matching_legs = [
                {"flight_leg_index": leg[0]["flight_leg_index"], "start_utc": utc(leg[0]["timestamp_unix_s"]),
                 "end_utc": utc(leg[-1]["timestamp_unix_s"]), "points": len(leg)}
                for leg in segments(points, 120)
                if onset is not None and leg[0]["timestamp_unix_s"] <= onset <= leg[-1]["timestamp_unix_s"]
            ]
            result = {
                "schema_version": 1, "date_utc": key[0], "icao24": key[1],
                "trace_file": record["trace_file"], "raw_signal_count": len(signals),
                "signals": signals, "decision": decision, "decision_reason": reason,
                "review_status": "human_reviewed" if review else "automated_trace_audit",
                "reviewer": review.get("reviewer") if review else None,
                "usable_onset_unix_s": onset if decision == "include_proxy" else None,
                "usable_onset_utc": utc(onset) if onset is not None and decision == "include_proxy" else None,
                "signal_leg_coverage": matching_legs,
                "point_count": len(points),
                "coverage_start_utc": utc(points[0]["timestamp_unix_s"]) if points else None,
                "coverage_end_utc": utc(points[-1]["timestamp_unix_s"]) if points else None,
            }
            counts[decision] += 1
            counts[result["review_status"]] += 1
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
    if set(reviews) - seen:
        raise ValueError(f"Reviews reference unknown candidates: {set(reviews) - seen}")
    summary = {"schema_version": 1, "candidate_count": len(seen), "counts": dict(counts),
               "signal_values": dict(sorted(signal_values.items())),
               "policy": "Exclude lifeguard-only priority status from the declaration proxy; human overrides require evidence.",
               "audit_file": str(output.resolve())}
    (output_dir / "candidate_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.collection_dir, args.output_dir, args.reviews), indent=2))


if __name__ == "__main__":
    main()
