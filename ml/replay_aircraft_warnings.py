"""Replay one development date through the per-aircraft warning I/O path."""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from detection.models import Aircraft, FlightState, Kinematics, Position
from ml.aircraft_warning import AircraftWarningEngine, Mode
from ml.event_supervised import load_compact_dataset
from ml.positive_pattern_pilot import COLLECTION, DATASET, ROOT, event_catalog
from scripts.fetch_event_traces import load_raw_trace, trace_points

REPLAY_DATE = "2025-11-15"
OUTPUT = ROOT / "data/learning/ten_day_development/aircraft_warning_replay"
MODES: tuple[Mode, ...] = ("speed_loss", "altitude_reversal", "combined")
MAX_PREDECLARATION_LEAD_SECONDS = 1200.0


def trace_state(point: dict[str, Any]) -> FlightState:
    """Convert one normalized historical point to the public FlightState I/O."""
    return FlightState(
        timestamp=float(point["timestamp_unix_s"]),
        icao24=str(point["icao24"]).lower(),
        flight_id=str(point.get("callsign") or point["icao24"]).strip() or str(point["icao24"]),
        aircraft=Aircraft(registration=point.get("registration"), type_code=point.get("aircraft_type")),
        position=Position(
            latitude=point.get("latitude_deg"), longitude=point.get("longitude_deg"),
            altitude_baro_ft=point.get("altitude_baro_ft"), altitude_geom_ft=point.get("altitude_geom_ft"),
            on_ground=point.get("on_ground"), source=point.get("source"),
            stale=bool((point.get("trace_flags") or 0) & 1),
        ),
        kinematics=Kinematics(
            ground_speed_kts=point.get("ground_speed_kt"), track_deg=point.get("track_deg"),
            vertical_rate_baro_fpm=point.get("vertical_rate_fpm"),
            vertical_rate_geom_fpm=point.get("vertical_rate_geom_fpm"),
        ),
        origin="history",
    )


def records_for_date(collection: Path, date: str) -> list[tuple[str, dict[str, Any]]]:
    records = []
    for cohort, directory in (("candidate", "emergency_candidates"), ("control_sample", "control_manifests")):
        path = collection / directory / f"{date}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            records.append((cohort, record))
    return sorted(records, key=lambda item: (item[0], item[1]["icao24"]))


def observed_airborne_seconds(points: list[dict[str, Any]]) -> float:
    return sum(
        current["timestamp_unix_s"] - previous["timestamp_unix_s"]
        for previous, current in zip(points, points[1:])
        if previous.get("on_ground") is False and current.get("on_ground") is False
        and 0 < current["timestamp_unix_s"] - previous["timestamp_unix_s"] <= 120
        and previous.get("flight_leg_index") == current.get("flight_leg_index")
    )


def rolling_hour_peak(timestamps: list[float]) -> int:
    ordered = sorted(timestamps)
    return max((index - bisect.bisect_right(ordered, timestamp - 3600) + 1
                for index, timestamp in enumerate(ordered)), default=0)


def summarize_mode(
    alerts: list[dict[str, Any]], events: dict[str, dict[str, Any]], control_hours: float,
) -> dict[str, Any]:
    event_hits: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    events_by_aircraft: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events.values():
        events_by_aircraft[event["icao24"]].append(event)
    control_alerts = []
    candidate_unmatched = 0
    for alert in alerts:
        if alert["source_cohort"] == "control_sample":
            control_alerts.append(alert)
            continue
        linked = [event for event in events_by_aircraft[alert["icao24"]]
                  if 0 < event["event_unix_s"] - alert["timestamp"] <= MAX_PREDECLARATION_LEAD_SECONDS]
        if not linked:
            candidate_unmatched += 1
        for event in linked:
            lead = event["event_unix_s"] - alert["timestamp"]
            event_hits[event["event_id"]].append((lead, alert))
    detected = {}
    band_counts = {"early_10_to_20_minutes_only": 0, "target_2_to_10_minutes": 0,
                   "late_under_2_minutes_only": 0}
    target_leads = []
    for event_id, hits in event_hits.items():
        early = [item for item in hits if item[0] > 600]
        target = [item for item in hits if 120 < item[0] <= 600]
        late = [item for item in hits if item[0] <= 120]
        if target:
            band = "target_2_to_10_minutes"
            target_leads.append(max(item[0] for item in target))
        elif early:
            band = "early_10_to_20_minutes_only"
        else:
            band = "late_under_2_minutes_only"
        band_counts[band] += 1
        first_lead, first_alert = max(hits, key=lambda item: item[0])
        detected[event_id] = {
            **events[event_id], "warning_seconds": first_lead,
            "alert_timestamp": first_alert["timestamp"], "signals": first_alert["signals"],
            "warning_band": band,
            "all_predeclaration_warning_seconds": sorted((lead for lead, _ in hits), reverse=True),
        }
    warnings = [event["warning_seconds"] for event in detected.values()]
    timestamps = [alert["timestamp"] for alert in control_alerts]
    return {
        "event_count": len(events), "detected_events": len(detected),
        "detected_aircraft": len({event["icao24"] for event in detected.values()}),
        "event_recall": len(detected) / len(events) if events else None,
        "median_warning_seconds": statistics.median(warnings) if warnings else None,
        "warning_band_counts": band_counts,
        "strict_2_to_10_minute_events": band_counts["target_2_to_10_minutes"],
        "strict_2_to_10_minute_median_warning_seconds": statistics.median(target_leads) if target_leads else None,
        "control_observed_airborne_hours": control_hours,
        "control_false_alerts": len(control_alerts),
        "control_false_alerts_per_1000_flight_hours": len(control_alerts) * 1000 / control_hours if control_hours else None,
        "control_false_alerts_per_24_clock_hours": len(control_alerts) / 24,
        "control_false_alerts_peak_rolling_hour": rolling_hour_peak(timestamps),
        "candidate_alerts_outside_20_minute_predeclaration_window": candidate_unmatched,
        "matched_events": sorted(detected.values(), key=lambda event: event["event_id"]),
        "missed_events": sorted((event for event_id, event in events.items() if event_id not in detected),
                                key=lambda event: event["event_id"]),
        "control_alert_examples": control_alerts[:20],
    }


def run_replay(
    date: str = REPLAY_DATE, collection: Path = COLLECTION,
    dataset: Path = DATASET, output_dir: Path = OUTPUT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, labeled_rows = load_compact_dataset(dataset, dataset.parent / "event_supervised/compact_episode_cache.joblib")
    indices = [index for index, row in enumerate(labeled_rows) if row["date_utc"] == date]
    events = event_catalog(labeled_rows, indices)
    if not events:
        raise ValueError(f"No eligible declaration events for {date}")
    alerts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    control_seconds = 0.0
    prediction_path = output_dir / f"predictions_{date}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for cohort, record in records_for_date(collection, date):
            path = collection / record["trace_file"]
            raw = load_raw_trace(path)
            points = sorted(trace_points(raw, record["icao24"]),
                            key=lambda point: (point["timestamp_unix_s"], point["point_index"]))
            if cohort == "control_sample":
                control_seconds += observed_airborne_seconds(points)
            engines = {mode: AircraftWarningEngine(mode) for mode in MODES}
            previous_leg = None
            for point in points:
                leg = point["flight_leg_index"]
                if previous_leg is not None and leg != previous_leg:
                    for engine in engines.values():
                        engine.reset(record["icao24"])
                previous_leg = leg
                state = trace_state(point)
                for mode, engine in engines.items():
                    result = engine.update(state)
                    counts[mode] += 1
                    if result.evaluated or result.alert:
                        mapping = {"mode": mode, "source_cohort": cohort, **result.to_mapping()}
                        stream.write(json.dumps(mapping, separators=(", ", ": "), allow_nan=False) + "\n")
                    if result.alert:
                        alerts[mode].append({
                            "date_utc": date, "icao24": result.icao24,
                            "timestamp": result.timestamp, "source_cohort": cohort,
                            "signals": [signal.to_mapping() for signal in result.signals],
                        })
    control_hours = control_seconds / 3600.0
    summaries = {mode: summarize_mode(alerts[mode], events, control_hours) for mode in MODES}
    combined = summaries["combined"]
    report = {
        "schema_version": 2, "date_utc": date,
        "target": "any warning in the 20 minutes before an observed ADS-B declaration",
        "warning_timing_policy": (
            "Count a warning at any positive lead up to 20 minutes; report >10 minutes, "
            "2–10 minutes, and <2 minutes separately. Alerts farther away are not attributed "
            "to the declaration. This is an exploratory association, not confirmed causation."),
        "input": "one protocol FlightState per observation per aircraft",
        "output": "one PredictionResult per update; evaluated results saved to prediction JSONL",
        "modes": summaries,
        "updates_processed_per_mode": dict(counts),
        "prediction_file": prediction_path.name,
        "acceptance": {
            "maximum_control_false_alerts_in_any_rolling_hour": 60,
            "combined_adds_events_over_both_single_rules": combined["detected_events"] > max(
                summaries["speed_loss"]["detected_events"],
                summaries["altitude_reversal"]["detected_events"]),
            "combined_below_hourly_limit": combined["control_false_alerts_peak_rolling_hour"] < 60,
        },
        "warning": "Development replay on a sampled fleet. Prior exploration used these dates; no future-date or full-fleet claim.",
    }
    report["acceptance"]["passed"] = all(
        report["acceptance"][key]
        for key in ("combined_adds_events_over_both_single_rules", "combined_below_hourly_limit")
    )
    (output_dir / f"report_{date}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=REPLAY_DATE)
    parser.add_argument("--collection-dir", type=Path, default=COLLECTION)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run_replay(args.date, args.collection_dir, args.dataset, args.output_dir)
    print(json.dumps({"date": report["date_utc"], "passed": report["acceptance"]["passed"],
                      "modes": {name: {
                          "events": result["detected_events"],
                          "control_alerts": result["control_false_alerts"],
                          "rolling_hour_peak": result["control_false_alerts_peak_rolling_hour"],
                      } for name, result in report["modes"].items()}}))
