"""Persistent, bounded cache for adsb.lol historical replay days.

Pinned days live in ``fixed``; up to MAX_CACHED_DAYS downloaded days live in
``cache`` and rotate by least-recently-used order. The archive root defaults to
``data/archive`` and can be moved with SYREN_ARCHIVE_DIR.
"""

import argparse
import datetime
import json
import os
import shutil
import tarfile
import threading
import urllib.error
import urllib.parse
import urllib.request

ARCHIVE_DIR = os.environ.get(
    "SYREN_ARCHIVE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive"),
)
FIXED_DAYS = ["2024-07-20", "2026-09-25"]
MAX_CACHED_DAYS = 7
MIN_DOWNLOAD_FREE_BYTES = 5 * 1024**3
MIN_FREE_RESERVE_BYTES = 512 * 1024**2
FIRST_DAY = "2023-02-16" 
REPO_YEARS = range(2023, 2027) 
RELEASE_API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
PODS = ["prod-0", "prod-1", "staging-0", "test-0"]   
USER_AGENT = "syren-hackathon"
REPORT_EVERY_BYTES = 100_000_000

_swap_lock = threading.Lock()
_swap_status = {"state": "idle", "date": None, "done_bytes": 0, "total_bytes": 0, "error": None}
_migration_lock = threading.Lock()
_migrated_roots = set()


# Describe where urllib will send a request without exposing proxy credentials.
def _request_context(url):
    parsed = urllib.parse.urlsplit(url)
    proxy = urllib.request.getproxies().get(parsed.scheme)
    if proxy:
        proxy_url = urllib.parse.urlsplit(proxy if "://" in proxy else "//" + proxy)
        route = f"proxy={proxy_url.hostname or 'configured'}"
    else:
        route = "proxy=none"
    return f"host={parsed.hostname or 'unknown'} {route}"


# GET a URL (the GitHub API) and return the parsed JSON body.
def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"history release lookup failed "
            f"(at={datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}, "
            f"{_request_context(url)}): "
            f"{type(error).__name__}: {error}"
        ) from error


# (first, last) day in the archive as "YYYY-MM-DD", e.g. for a date picker's limits.
# First is FIRST_DAY, last is yesterday in UTC.
def available_range():
    yesterday = datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=1)
    return FIRST_DAY, yesterday.isoformat()


# Raise ValueError unless date is a real "YYYY-MM-DD" day inside available_range().
def _check_date(date):
    first, last = available_range()
    if not isinstance(date, str):
        raise ValueError("history date must be a YYYY-MM-DD string")
    if datetime.date.fromisoformat(date).isoformat() != date or not first <= date <= last:
        raise ValueError(f"{date!r} is not a day in the history archive ({first} to {last})")


def _day_directories(folder):
    if not os.path.isdir(folder):
        return []
    result = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        try:
            if datetime.date.fromisoformat(name).isoformat() == name:
                result.append(name)
        except ValueError:
            continue
    return result


# Look up one day's release on GitHub and return [(url, size), ...] of its tar parts, in order.
# Tries each pod in PODS, in that year's repo and then the previous year's.
# Raises LookupError if no release exists for the day.
def find_release(date):
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


# A file-like object that reads the tar parts back to back as if they were one file,
# so tarfile can stream a split archive without saving the parts to disk.
class _Downloads:

    # Remember the part URLs and total size; nothing downloads until read() is called.
    def __init__(self, parts, on_progress=None, date=None):
        self._urls = [url for url, _ in parts]
        self._response = None
        self._active_url = None
        self._active_part = 0
        self._date = date
        self._on_progress = on_progress
        self.total = sum(size for _, size in parts)
        self.done = 0
        self._next_report = 0

    # Return the next chunk of bytes, opening the next part when one runs out,
    # and b"" once every part is read. Reports progress as it goes.
    def read(self, size=-1):
        while True:
            if self._response is None:
                if not self._urls:
                    return b""
                self._active_url = self._urls.pop(0)
                self._active_part += 1
                request = urllib.request.Request(self._active_url, headers={"User-Agent": USER_AGENT})
                try:
                    self._response = urllib.request.urlopen(request, timeout=60)
                except Exception as error:
                    raise RuntimeError(
                        f"history asset open failed "
                        f"(at={datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}, "
                        f"date={self._date}, "
                        f"part={self._active_part}, {_request_context(self._active_url)}): "
                        f"{type(error).__name__}: {error}"
                    ) from error
            try:
                chunk = self._response.read(size)
            except Exception as error:
                raise RuntimeError(
                    f"history asset read failed "
                    f"(at={datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}, "
                    f"date={self._date}, part={self._active_part}, "
                    f"{_request_context(self._active_url)}): {type(error).__name__}: {error}"
                ) from error
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


# Stream a day's tar parts, keep only the trace_full_* files, and move them to final.
# Works in root/.partial so an interrupted download never looks like a finished day.
def _download(date, parts, final, root, on_progress=None):
    os.makedirs(root, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < MIN_DOWNLOAD_FREE_BYTES:
        raise OSError(
            f"history volume has only {free / 1024**3:.1f} GiB free; "
            f"at least {MIN_DOWNLOAD_FREE_BYTES / 1024**3:.0f} GiB is required to start a day download"
        )

    partial = os.path.join(root, ".partial", date)
    shutil.rmtree(partial, ignore_errors=True)  # left over from an interrupted download
    print(f"{date}: {sum(size for _, size in parts) / 1e9:.1f} GB in {len(parts)} part(s)")

    count = 0
    extracted_bytes = 0
    next_space_check = 0
    os.makedirs(partial, exist_ok=True)
    try:
        with tarfile.open(fileobj=_Downloads(parts, on_progress, date=date), mode="r|") as tar:
            for member in tar:
                name = os.path.basename(member.name)
                if not (member.isfile() and name.startswith("trace_full_")):
                    continue
                if extracted_bytes >= next_space_check:
                    free = shutil.disk_usage(root).free
                    if free < member.size + MIN_FREE_RESERVE_BYTES:
                        raise OSError(
                            f"not enough free space to safely extract {date}; "
                            f"{free / 1024**3:.1f} GiB remains"
                        )
                    next_space_check = extracted_bytes + 256 * 1024**2
                icao24 = name[len("trace_full_"):-len(".json")]
                folder = os.path.join(partial, "traces", icao24[-2:])
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, name), "wb") as out:
                    shutil.copyfileobj(tar.extractfile(member), out)
                extracted_bytes += member.size
                count += 1

        os.makedirs(os.path.dirname(final), exist_ok=True)
        os.rename(partial, final)
    finally:
        if os.path.isdir(partial):
            shutil.rmtree(partial, ignore_errors=True)
    print(f"\n{count} aircraft traces -> {final}")
    return final


# Move the old single-day swap directory into the multi-day cache once per root.
def _migrate_legacy_swap(root):
    normalized_root = os.path.abspath(root)
    if normalized_root in _migrated_roots:
        return
    with _migration_lock:
        if normalized_root in _migrated_roots:
            return
        legacy = os.path.join(root, "swap")
        if os.path.isdir(legacy):
            cache = os.path.join(root, "cache")
            os.makedirs(cache, exist_ok=True)
            for date in os.listdir(legacy):
                source = os.path.join(legacy, date)
                if not os.path.isdir(source):
                    continue
                pinned = os.path.join(root, "fixed", date)
                target = os.path.join(cache, date)
                if os.path.isdir(pinned) or os.path.exists(target):
                    shutil.rmtree(source)
                else:
                    os.replace(source, target)
            try:
                os.rmdir(legacy)
            except OSError:
                pass
        # Older builds kept all replay exports directly under the archive root.
        legacy_exports = os.path.join(root, "exports")
        if os.path.isdir(legacy_exports):
            for name in os.listdir(legacy_exports):
                date = name[:10]
                if len(name) < 12 or name[10] != "_":
                    continue
                day = next((
                    os.path.join(root, kind, date)
                    for kind in ("fixed", "cache", "swap")
                    if os.path.isdir(os.path.join(root, kind, date))
                ), None)
                if day is None:
                    continue
                target_dir = os.path.join(day, "exports")
                os.makedirs(target_dir, exist_ok=True)
                target = os.path.join(target_dir, name)
                source = os.path.join(legacy_exports, name)
                if os.path.exists(target):
                    os.remove(source)
                else:
                    os.replace(source, target)
            try:
                os.rmdir(legacy_exports)
            except OSError:
                pass
        # A partial download is never a usable replay day; discard leftovers from older runs.
        shutil.rmtree(os.path.join(root, ".partial"), ignore_errors=True)
        _migrated_roots.add(normalized_root)


# The folder holding a day's traces (fixed or cached), or None if it isn't on disk.
def day_dir(date, root=ARCHIVE_DIR):
    _migrate_legacy_swap(root)
    for kind in ("fixed", "cache", "swap"):
        path = os.path.join(root, kind, date)
        if os.path.isdir(path):
            return path
    return None


# Path of one aircraft's trace file on a day on disk, or None if the day or aircraft isn't there.
def trace_path(date, icao24, root=ARCHIVE_DIR):
    folder = day_dir(date, root)
    if folder is None:
        return None
    icao24 = icao24.lower()
    path = os.path.join(folder, "traces", icao24[-2:], f"trace_full_{icao24}.json")
    return path if os.path.exists(path) else None


# The days on disk. ``swap`` remains as an alias for the most recently used
# cached day for compatibility with the previous one-swap API.
def days_on_disk(root=ARCHIVE_DIR):
    _migrate_legacy_swap(root)
    fixed_folder = os.path.join(root, "fixed")
    cache_folder = os.path.join(root, "cache")
    fixed = sorted(_day_directories(fixed_folder))
    cached = sorted(
        _day_directories(cache_folder)
    )
    most_recent = max(
        cached,
        key=lambda date: os.path.getmtime(os.path.join(cache_folder, date)),
        default=None,
    )
    return {"fixed": fixed, "cache": cached, "swap": most_recent}


def touch_cached_day(date, root=ARCHIVE_DIR):
    """Mark a cached day recently used; pinned days are deliberately unchanged."""
    path = os.path.join(root, "cache", date)
    if os.path.isdir(path):
        os.utime(path, None)
        return True
    return False


def trim_cache(root=ARCHIVE_DIR, protected=()):
    """Keep only the newest cache days, excluding a day being replayed."""
    _migrate_legacy_swap(root)
    cache_folder = os.path.join(root, "cache")
    if not os.path.isdir(cache_folder):
        return []
    protected = set(protected)
    entries = _day_directories(cache_folder)
    entries.sort(key=lambda date: (os.path.getmtime(os.path.join(cache_folder, date)), date))
    removed = []
    while len(entries) > MAX_CACHED_DAYS:
        victim = next((date for date in entries if date not in protected), None)
        if victim is None:
            break
        shutil.rmtree(os.path.join(cache_folder, victim))
        entries.remove(victim)
        removed.append(victim)
        print(f"evicted least-recently-used history day {victim}")
    return removed


# Make the fixed days on disk match days: download the missing ones and delete
# the ones no longer listed. Run once per machine with `python3 -m data.history setup`.
def setup_fixed_days(days=FIXED_DAYS, root=ARCHIVE_DIR, find=find_release):
    _migrate_legacy_swap(root)
    for date in days_on_disk(root)["fixed"]:
        if date not in days:
            shutil.rmtree(os.path.join(root, "fixed", date))
            print(f"removed {date}, no longer a fixed day")
    for date in days:
        fixed = os.path.join(root, "fixed", date)
        if os.path.isdir(fixed):
            continue
        cached = os.path.join(root, "cache", date)
        if os.path.isdir(cached):
            os.makedirs(os.path.dirname(fixed), exist_ok=True)
            os.replace(cached, fixed)
        else:
            _download(date, find(date), fixed, root)
    trim_cache(root)


# Download one day into the rotating cache. A failed download leaves cached days intact.
def _swap(date, root, find, on_progress, trim=True):
    _migrate_legacy_swap(root)
    _check_date(date)  # raises ValueError for a malformed date
    existing = day_dir(date, root)
    if existing is not None:
        touch_cached_day(date, root)
        return existing
    parts = find(date)
    final = os.path.join(root, "cache", date)
    downloaded = _download(date, parts, final, root, on_progress)
    touch_cached_day(date, root)
    if trim:
        trim_cache(root, protected={date})
    return downloaded


# Make date available, replacing the previous swap day. Blocks until done (~10 min).
# Raises RuntimeError if another swap is already running.
def swap_day(date, root=ARCHIVE_DIR, find=find_release, on_progress=None):
    if not _swap_lock.acquire(blocking=False):
        raise RuntimeError(f"already downloading {_swap_status['date']}")
    try:
        return _swap(date, root, find, on_progress)
    finally:
        _swap_lock.release()


# What the frontend calls: start swap_day in a background thread and return at once.
# Follow it with swap_status(). Raises RuntimeError if a swap is already running.
def start_swap(date, root=ARCHIVE_DIR, find=find_release):
    if not _swap_lock.acquire(blocking=False):
        raise RuntimeError(f"already downloading {_swap_status['date']}")
    _swap_status.update(state="downloading", date=date, done_bytes=0, total_bytes=0, error=None)

    # Called as bytes arrive; updates the numbers swap_status() returns.
    def on_progress(done, total):
        _swap_status.update(done_bytes=done, total_bytes=total)

    # The background thread: do the swap, record ready or failed, then free the lock.
    def run():
        try:
            _swap(date, root, find, on_progress)
            _swap_status["state"] = "ready"
        except Exception as error:
            _swap_status.update(state="failed", error=str(error))
        finally:
            _swap_lock.release()

    threading.Thread(target=run, daemon=True).start()


# Progress of the last start_swap, for the frontend to poll:
# {"state": "idle"|"downloading"|"ready"|"failed", "date", "done_bytes", "total_bytes", "error"}
def swap_status():
    return dict(_swap_status)


# Command line: setup | swap DATE | list (see the top of this file).
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage adsb.lol history days on disk.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="download FIXED_DAYS and remove fixed days no longer listed")
    cache_command = commands.add_parser("cache", aliases=["swap"], help="download or reuse a history day")
    cache_command.add_argument("date", help="YYYY-MM-DD")
    commands.add_parser("list", help="show the days on disk")
    args = parser.parse_args()

    if args.command == "setup":
        setup_fixed_days()
    elif args.command in ("cache", "swap"):
        swap_day(args.date)
    elif args.command == "list":
        days = days_on_disk()
        print(f"archive  {' to '.join(available_range())}")
        print(f"fixed  {', '.join(days['fixed']) or '-'}")
        print(f"cache  {', '.join(days['cache']) or '-'}")
