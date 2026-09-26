#!/usr/bin/env python3
"""Collect whole-day ADSB.lol archives and identify explicit emergency signals.

The collector streams GitHub's daily tar archive without extracting its members.
It retains all aircraft-day traces with an observed emergency field or emergency
squawk, plus a deterministic random sample of other traces for later review.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "collection"
RAW_BASE = "https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt"
USER_AGENT = "Syren-Research-Prototype/0.2 (historical ADS-B collection)"
EMERGENCY_SQUAWKS = {"7500", "7600", "7700"}
EMERGENCY_NONE = {"", "none", "noemergency", "no emergency", "0", "null"}
TRACE_NAME = re.compile(r"(?:^|/)trace_full_([0-9a-fA-F]{6})\.json(?:\.gz)?$")
DATE_IN_RELEASE = re.compile(r"v(\d{4})\.(\d{2})\.(\d{2})-")
ACTIVE_EMERGENCY_MARKER = re.compile(rb'"emergency"\s*:\s*"(?!none\s*")', re.IGNORECASE)
EMERGENCY_SQUAWK_MARKER = re.compile(rb'"squawk"\s*:\s*"?0*(?:7500|7600|7700)"?(?=\s*[,}])')


class CollectionError(Exception):
    """Invalid release metadata or archive content."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True, help="UTC dates, YYYY-MM-DD")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--controls-per-day", type=int, default=100)
    parser.add_argument("--max-archive-bytes", type=int, default=6 * 1024**3)
    parser.add_argument("--keep-archives", action="store_true", help="keep downloaded tar archives")
    parser.add_argument("--retry-count", type=int, default=3)
    return parser.parse_args()


def fetch(url: str, *, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise CollectionError(f"HTTP {response.status}: {url}")
        return response.read()


def get_release_urls(day: date, retries: int = 3) -> list[str]:
    url = RAW_BASE.format(year=day.year)
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            manifest = fetch(url).decode("utf-8")
            for line in manifest.splitlines():
                urls = [item.strip() for item in line.split(",") if item.strip()]
                if not urls:
                    continue
                match = DATE_IN_RELEASE.search(urls[0])
                if match and date(*(int(part) for part in match.groups())) == day:
                    return urls
            raise CollectionError(f"No preferred release found for {day} in {url}")
        except (OSError, UnicodeError, urllib.error.URLError, CollectionError) as exc:
            error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 10))
    raise CollectionError(f"Could not resolve release for {day}: {error}")


def download_archive(urls: list[str], destination: Path, max_bytes: int, retries: int) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as out:
            for url in urls:
                part_start = out.tell()
                for attempt in range(retries + 1):
                    try:
                        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                        with urllib.request.urlopen(request, timeout=120) as response:
                            if response.status != 200:
                                raise CollectionError(f"HTTP {response.status}: {url}")
                            while True:
                                block = response.read(4 * 1024 * 1024)
                                if not block:
                                    break
                                size += len(block)
                                if size > max_bytes:
                                    raise CollectionError(
                                        f"Archive exceeded --max-archive-bytes ({max_bytes:,}); partial file removed"
                                    )
                                out.write(block)
                        break
                    except (OSError, urllib.error.URLError, CollectionError) as exc:
                        if isinstance(exc, CollectionError) and "exceeded" in str(exc):
                            raise
                        out.seek(part_start)
                        out.truncate()
                        size = part_start
                        if attempt == retries:
                            raise CollectionError(f"Download failed for {url}: {exc}") from exc
                        time.sleep(min(2**attempt, 10))
        digest = hashlib.sha256()
        with destination.open("rb") as archive:
            for block in iter(lambda: archive.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return size, digest.hexdigest()
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def inspect_trace(payload: bytes, *, compressed: bool = True) -> dict[str, Any] | None:
    """Parse one trace and return metadata plus any explicit emergency signals."""
    try:
        raw = gzip.decompress(payload) if compressed and payload[:2] == b"\x1f\x8b" else payload
        trace = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    points = trace.get("trace") if isinstance(trace, dict) else None
    if not isinstance(points, list):
        return None

    emergency: Any = None
    squawk: Any = None
    callsign = trace.get("flight")
    onset: list[dict[str, Any]] = []
    prior_emergency: str | None = None
    prior_squawk: str | None = None
    valid_positions = 0
    for point in points:
        if not isinstance(point, list) or len(point) < 8:
            continue
        valid_positions += 1
        extra = point[8] if len(point) > 8 and isinstance(point[8], dict) else None
        if extra:
            if "emergency" in extra:
                emergency = extra["emergency"]
            if "squawk" in extra:
                squawk = extra["squawk"]
            if extra.get("flight"):
                callsign = str(extra["flight"]).strip()
        emergency_text = str(emergency).strip().lower() if emergency is not None else ""
        squawk_text = str(squawk).strip().zfill(4) if squawk is not None else ""
        active_emergency = emergency_text not in EMERGENCY_NONE
        active_squawk = squawk_text in EMERGENCY_SQUAWKS
        if active_emergency and emergency_text != prior_emergency:
            onset.append({"offset_s": point[0], "signal": "emergency_field", "value": emergency_text})
        if active_squawk and squawk_text != prior_squawk:
            onset.append({"offset_s": point[0], "signal": "emergency_squawk", "value": squawk_text})
        prior_emergency = emergency_text if active_emergency else None
        prior_squawk = squawk_text if active_squawk else None
    base_timestamp = trace.get("timestamp")
    if isinstance(base_timestamp, (int, float)) and not isinstance(base_timestamp, bool):
        for signal in onset:
            offset = signal.get("offset_s")
            if isinstance(offset, (int, float)) and not isinstance(offset, bool):
                signal["timestamp_utc"] = datetime.fromtimestamp(
                    base_timestamp + offset, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z")
    return {
        "icao24": str(trace.get("icao", trace.get("hex", ""))).lower() or None,
        "registration": trace.get("r"),
        "aircraft_type": trace.get("t"),
        "callsign": callsign,
        "trace_timestamp_unix_s": trace.get("timestamp"),
        "position_count": valid_positions,
        "signals": onset,
    }


def scan_archive(
    archive_path: Path,
    day: date,
    out_dir: Path,
    controls_per_day: int,
    source_urls: list[str],
) -> dict[str, Any]:
    day_key = day.isoformat()
    trace_root = out_dir / "traces" / day_key
    positives: list[dict[str, Any]] = []
    rng = random.Random(day_key)
    controls_seen = 0
    controls: list[tuple[int, str, bytes, dict[str, Any]]] = []
    aircraft_seen = 0
    invalid = 0
    try:
        with tarfile.open(archive_path, mode="r|") as archive:
            for member in archive:
                match = TRACE_NAME.search(member.name)
                if not match or not member.isfile():
                    continue
                aircraft_seen += 1
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    invalid += 1
                    continue
                payload = file_obj.read()
                identity_hex = match.group(1).lower()
                try:
                    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
                except OSError:
                    invalid += 1
                    continue
                if not raw.lstrip().startswith(b"{") or b'"trace"' not in raw:
                    invalid += 1
                    continue
                possible_signal = bool(
                    ACTIVE_EMERGENCY_MARKER.search(raw) or EMERGENCY_SQUAWK_MARKER.search(raw)
                )
                if possible_signal:
                    info = inspect_trace(raw, compressed=False)
                    if info is None:
                        invalid += 1
                        continue
                    signals = info.pop("signals")
                else:
                    info = None
                    signals = []
                if signals:
                    assert info is not None
                    if not info["icao24"]:
                        info["icao24"] = identity_hex
                    relative = Path("traces") / day_key / "emergency" / f"{identity_hex}.json.gz"
                    target = out_dir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    positives.append({
                        "date_utc": day_key,
                        "label": "observed_adsb_emergency_signal",
                        "label_strength": "explicit_transponder_indicator",
                        "source": "adsb.lol daily historical archive",
                        "archive_member": member.name,
                        "source_archive_urls": source_urls,
                        "trace_file": relative.as_posix(),
                        **info,
                        "signals": signals,
                    })
                    continue
                controls_seen += 1
                if controls_per_day <= 0:
                    continue
                # Reservoir sampling gives every non-trigger aircraft trace equal probability.
                item = (controls_seen, identity_hex, payload, info)
                if len(controls) < controls_per_day:
                    controls.append(item)
                else:
                    slot = rng.randrange(controls_seen)
                    if slot < controls_per_day:
                        controls[slot] = item
    except (OSError, tarfile.TarError) as exc:
        raise CollectionError(f"Could not scan {archive_path}: {exc}") from exc

    control_records: list[dict[str, Any]] = []
    control_dir = trace_root / "control_sample"
    control_dir.mkdir(parents=True, exist_ok=True)
    for _, identity_hex, payload, info in sorted(controls, key=lambda item: item[1]):
        info = inspect_trace(payload)
        if info is None or info["position_count"] == 0 or info["signals"]:
            invalid += 1
            continue
        info.pop("signals")
        relative = Path("traces") / day_key / "control_sample" / f"{identity_hex}.json.gz"
        (out_dir / relative).write_bytes(payload)
        control_records.append({
            "date_utc": day_key,
            "label": "unlabeled_no_observed_declaration",
            "source": "adsb.lol daily historical archive",
            "trace_file": relative.as_posix(),
            **info,
        })
    return {
        "date_utc": day_key,
        "aircraft_trace_files_scanned": aircraft_seen,
        "aircraft_with_explicit_emergency_signal": len(positives),
        "non_trigger_trace_files_considered": controls_seen,
        "control_traces_retained": len(control_records),
        "invalid_trace_files": invalid,
        "emergency_candidates": positives,
        "controls": control_records,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def collect_day(day: date, args: argparse.Namespace) -> dict[str, Any]:
    day_root = args.out_dir / "archives"
    archive_path = day_root / f"{day.isoformat()}.tar"
    urls = get_release_urls(day, args.retry_count)
    print(f"{day}: downloading {len(urls)} archive part(s)", flush=True)
    size, sha256 = download_archive(urls, archive_path, args.max_archive_bytes, args.retry_count)
    try:
        print(f"{day}: downloaded {size / 1024**3:.2f} GiB; scanning traces", flush=True)
        report = scan_archive(archive_path, day, args.out_dir, args.controls_per_day, urls)
        report.update({"archive_bytes": size, "archive_sha256": sha256, "archive_urls": urls})
        return report
    finally:
        if not args.keep_archives:
            archive_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.controls_per_day < 0 or args.max_archive_bytes <= 0 or args.retry_count < 0:
        print("control count and retry count must be non-negative; archive limit must be positive", file=sys.stderr)
        return 2
    try:
        dates = sorted({date.fromisoformat(value) for value in args.dates})
    except ValueError as exc:
        print(f"Invalid date: {exc}", file=sys.stderr)
        return 2
    if not dates:
        print("At least one date is required", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.out_dir / "reports"
    candidate_dir = args.out_dir / "emergency_candidates"
    control_manifest_dir = args.out_dir / "control_manifests"
    failures = 0
    for day in dates:
        try:
            report = collect_day(day, args)
            candidates = report.pop("emergency_candidates")
            controls = report.pop("controls")
            report_dir.mkdir(parents=True, exist_ok=True)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            control_manifest_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"{day.isoformat()}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            write_jsonl(candidate_dir / f"{day.isoformat()}.jsonl", candidates)
            write_jsonl(control_manifest_dir / f"{day.isoformat()}.jsonl", controls)
            print(
                f"{day}: scanned {report['aircraft_trace_files_scanned']:,}, "
                f"emergency candidates {report['aircraft_with_explicit_emergency_signal']}, "
                f"controls retained {report['control_traces_retained']}",
                flush=True,
            )
        except (OSError, CollectionError, urllib.error.URLError) as exc:
            failures += 1
            print(f"{day}: ERROR {exc}", file=sys.stderr, flush=True)
    print(f"Collection reports: {report_dir}; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
