"""Stage NTSB aviation CSV cases for verification, without creating training labels."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

NTSB_SEARCH = "https://www.ntsb.gov/Pages/AviationQueryV2.aspx"


def field(row: dict[str, str], *aliases: str) -> str:
    normalized = {"".join(c for c in key.lower() if c.isalnum()): value.strip()
                  for key, value in row.items() if key and isinstance(value, str)}
    return next((normalized.get("".join(c for c in alias.lower() if c.isalnum()), "")
                 for alias in aliases
                 if normalized.get("".join(c for c in alias.lower() if c.isalnum()), "")), "")


def parse_day(value: str) -> str:
    for pattern in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%Y %I:%M:%S %p",
                    "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse NTSB event date: {value!r}")


def discover(source: Path, output: Path, from_date: str = "2024-01-01") -> dict:
    date.fromisoformat(from_date)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("NTSB CSV has no header")
        candidates = []
        seen = set()
        for row in reader:
            event_id = field(row, "Accident Number", "AccidentNumber", "NTSB Number")
            date_text = field(row, "Event Date", "EventDate")
            registration = field(row, "Registration Number", "RegistrationNumber", "Registration").upper()
            event_type = field(row, "Investigation Type", "InvestigationType", "Event Type")
            if not event_id or not date_text or not registration:
                continue
            day = parse_day(date_text)
            if day < from_date or event_type.lower() not in {"accident", "incident"}:
                continue
            key = (event_id, registration)
            if key in seen:
                continue
            seen.add(key)
            source_url = field(row, "Investigation URL", "InvestigationURL", "Investigation Link") or NTSB_SEARCH
            candidates.append({
                "schema_version": 1, "source_agency": "NTSB", "source_dataset": NTSB_SEARCH,
                "source_record_url": source_url, "investigation_id": event_id,
                "event_date_local_or_unknown_tz": day, "investigation_type": event_type,
                "registration": registration, "aircraft_category": field(row, "Aircraft Category", "AircraftCategory"),
                "broad_phase_of_flight": field(row, "Broad Phase of Flight", "BroadPhaseOfFlight"),
                "verification_status": "pending_utc_time_icao_trace_and_event_type_review",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in candidates:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    return {"candidate_count": len(candidates), "output": str(output.resolve()),
            "warning": "Discovery rows are not training labels. Verify event identity, UTC time and precision, outcome, aircraft ICAO, and ADS-B coverage first."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntsb-csv", type=Path, required=True,
                        help="CSV exported from official NTSB Aviation Investigation Search")
    parser.add_argument("--output", type=Path, default=Path("data/learning/ntsb_event_candidates.jsonl"))
    parser.add_argument("--from-date", default="2024-01-01")
    args = parser.parse_args()
    print(json.dumps(discover(args.ntsb_csv, args.output, args.from_date), indent=2))


if __name__ == "__main__":
    main()
