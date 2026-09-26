"""adsb.lol history days on disk (github.com/adsblol/globe_history_YYYY).

Two kinds of day:
- FIXED_DAYS are hardcoded and always on disk (run `setup` once per machine).
- One "swap" day, picked from the frontend; picking another day replaces it.

Only the trace files are kept:
data/archive/{fixed,swap}/YYYY-MM-DD/traces/<last 2 hex>/trace_full_<icao24>.json

python3 -m data.history setup              # download FIXED_DAYS, remove fixed days no longer listed
python3 -m data.history swap 2026-09-20    # replace the swap day
python3 -m data.history list
"""

import argparse
import datetime
import json
import os
import shutil
import tarfile
import threading
import urllib.error
import urllib.request

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")
FIXED_DAYS = ["2026-09-24", "2026-09-25"]
FIRST_DAY = "2023-02-16" 
REPO_YEARS = range(2023, 2027) 
RELEASE_API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
PODS = ["prod-0", "prod-1", "staging-0", "test-0"]   
USER_AGENT = "syren-hackathon"
REPORT_EVERY_BYTES = 100_000_000

_swap_lock = threading.Lock()
_swap_status = {"state": "idle", "date": None, "done_bytes": 0, "total_bytes": 0, "error": None}


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def available_range():
    """(first, last) day in the archive as "YYYY-MM-DD", e.g. for a date picker's limits."""
    yesterday = datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=1)
    return FIRST_DAY, yesterday.isoformat()


def _check_date(date):
    """Raise ValueError unless date is a YYYY-MM-DD day inside available_range()."""
    first, last = available_range()
    if datetime.date.fromisoformat(date).isoformat() != date or not first <= date <= last:
        raise ValueError(f"{date!r} is not a day in the history archive ({first} to {last})")


def find_release(date):
    """[(url, size), ...] of the tar parts for one day, e.g. "2026-09-24", in order."""
    _check_date(date)
    year = int(date[:4])
    for pod in PODS:
        tag = f"v{date.replace('-', '.')}-planes-readsb-{pod}"
        for repo_year in (year, year - 1):  
            if repo_year not in REPO_YEARS:
                continue
            try:
                release = _get_json(RELEASE_API.format(year=repo_year, tag=tag))
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    continue
                raise
            assets = sorted((a for a in release["assets"] if ".tar" in a["name"]),
                            key=lambda a: a["name"])
            return [(a["browser_download_url"], a["size"]) for a in assets]
    raise LookupError(f"no adsb.lol history release found for {date}")


class _Downloads:
    """Reads several URLs back to back as if they were one file, reporting progress."""

    def __init__(self, parts, on_progress=None):
        self._urls = [url for url, _ in parts]
        self._response = None
        self._on_progress = on_progress
        self.total = sum(size for _, size in parts)
        self.done = 0
        self._next_report = 0

    def read(self, size=-1):
        while True:
            if self._response is None:
                if not self._urls:
                    return b""
                request = urllib.request.Request(self._urls.pop(0), headers={"User-Agent": USER_AGENT})
                self._response = urllib.request.urlopen(request, timeout=60)
            chunk = self._response.read(size)
            if chunk:
                self.done += len(chunk)
                if self._on_progress is not None:
                    self._on_progress(self.done, self.total)
                if self.done >= self._next_report:
                    print(f"\r  {self.done / 1e9:.2f} / {self.total / 1e9:.2f} GB", end="", flush=True)
                    self._next_report += REPORT_EVERY_BYTES
                return chunk
            self._response.close()
            self._response = None


def _download(date, parts, final, root, on_progress=None):
    """Stream a day's tar parts, keep only the trace files, and move them to final."""
    partial = os.path.join(root, ".partial")
    shutil.rmtree(partial, ignore_errors=True)  # left over from an interrupted download
    print(f"{date}: {sum(size for _, size in parts) / 1e9:.1f} GB in {len(parts)} part(s)")

    count = 0
    with tarfile.open(fileobj=_Downloads(parts, on_progress), mode="r|") as tar:
        for member in tar:
            name = os.path.basename(member.name)
            if not (member.isfile() and name.startswith("trace_full_")):
                continue
            icao24 = name[len("trace_full_"):-len(".json")]
            folder = os.path.join(partial, "traces", icao24[-2:])
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, name), "wb") as out:
                shutil.copyfileobj(tar.extractfile(member), out)
            count += 1

    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.rename(partial, final)
    print(f"\n{count} aircraft traces -> {final}")
    return final


def day_dir(date, root=ARCHIVE_DIR):
    """The folder holding a day's traces, or None if that day isn't on disk."""
    for kind in ("fixed", "swap"):
        path = os.path.join(root, kind, date)
        if os.path.isdir(path):
            return path
    return None


def trace_path(date, icao24, root=ARCHIVE_DIR):
    """Path of one aircraft's trace file for a day on disk, or None."""
    folder = day_dir(date, root)
    if folder is None:
        return None
    icao24 = icao24.lower()
    path = os.path.join(folder, "traces", icao24[-2:], f"trace_full_{icao24}.json")
    return path if os.path.exists(path) else None


def days_on_disk(root=ARCHIVE_DIR):
    """{"fixed": ["2026-09-24", ...], "swap": "2026-09-20" or None}: the days you can pick from."""
    fixed_folder = os.path.join(root, "fixed")
    swap_folder = os.path.join(root, "swap")
    fixed = sorted(os.listdir(fixed_folder)) if os.path.isdir(fixed_folder) else []
    swap = sorted(os.listdir(swap_folder)) if os.path.isdir(swap_folder) else []
    return {"fixed": fixed, "swap": swap[0] if swap else None}


def setup_fixed_days(days=FIXED_DAYS, root=ARCHIVE_DIR, find=find_release):
    """Download the fixed days that are missing and delete fixed days no longer listed."""
    for date in days_on_disk(root)["fixed"]:
        if date not in days:
            shutil.rmtree(os.path.join(root, "fixed", date))
            print(f"removed {date}, no longer a fixed day")
    for date in days:
        if day_dir(date, root) is None:
            _download(date, find(date), os.path.join(root, "fixed", date), root)


def _swap(date, root, find, on_progress):
    _check_date(date)  # raises ValueError for a malformed date
    existing = day_dir(date, root)
    if existing is not None:
        return existing  # a fixed day, or already the swap day
    parts = find(date)  # before deleting anything, so an unknown date keeps the old swap day
    shutil.rmtree(os.path.join(root, "swap"), ignore_errors=True)
    return _download(date, parts, os.path.join(root, "swap", date), root, on_progress)


def swap_day(date, root=ARCHIVE_DIR, find=find_release, on_progress=None):
    """Make date available, replacing the previous swap day. Blocks until done (~10 min)."""
    if not _swap_lock.acquire(blocking=False):
        raise RuntimeError(f"already downloading {_swap_status['date']}")
    try:
        return _swap(date, root, find, on_progress)
    finally:
        _swap_lock.release()


def start_swap(date, root=ARCHIVE_DIR, find=find_release):
    """Start swap_day in the background and return at once; follow it with swap_status()."""
    if not _swap_lock.acquire(blocking=False):
        raise RuntimeError(f"already downloading {_swap_status['date']}")
    _swap_status.update(state="downloading", date=date, done_bytes=0, total_bytes=0, error=None)

    def on_progress(done, total):
        _swap_status.update(done_bytes=done, total_bytes=total)

    def run():
        try:
            _swap(date, root, find, on_progress)
            _swap_status["state"] = "ready"
        except Exception as error:
            _swap_status.update(state="failed", error=str(error))
        finally:
            _swap_lock.release()

    threading.Thread(target=run, daemon=True).start()


def swap_status():
    """{"state": "idle"|"downloading"|"ready"|"failed", "date", "done_bytes", "total_bytes", "error"}"""
    return dict(_swap_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage adsb.lol history days on disk.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="download FIXED_DAYS and remove fixed days no longer listed")
    swap_command = commands.add_parser("swap", help="replace the swap day with another day")
    swap_command.add_argument("date", help="YYYY-MM-DD")
    commands.add_parser("list", help="show the days on disk")
    args = parser.parse_args()

    if args.command == "setup":
        setup_fixed_days()
    elif args.command == "swap":
        swap_day(args.date)
    elif args.command == "list":
        days = days_on_disk()
        print(f"archive  {' to '.join(available_range())}")
        print(f"fixed  {', '.join(days['fixed']) or '-'}")
        print(f"swap   {days['swap'] or '-'}")