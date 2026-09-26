import argparse
import bisect
import datetime
import json
import os
import re
import threading
import time

from data.history import ARCHIVE_DIR, day_dir, swap_day
from data.trace import export_day, open_jsonl

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "history.jsonl")
GONE_AFTER_S = 60  # drop an aircraft from the map after this long without a new state
# Lines written by make_flight_state + json.dumps start like {"timestamp": 1790265600.03, "icao24": "a1b2c3",
# so both can be read without parsing the whole line, which matters for millions of lines.
LINE_START = re.compile(r'\{"timestamp": ([^,]+), "icao24": "([^"]+)"')

_lock = threading.Lock()
_export_lock = threading.Lock()  # one export at a time; starting the same hour twice reuses it
_generation = 0  # bumped by every start and stop, so an old background thread knows to quit
_player = None
_status = {"state": "idle", "date": None, "clock": None, "first": None, "last": None,
           "done_bytes": 0, "total_bytes": 0, "export_done": 0, "export_total": 0, "error": None}


# (timestamp, icao24) of one FlightState JSON line.
def _time_and_icao24(line):
    match = LINE_START.match(line)
    if match:
        return float(match[1]), match[2]
    state = json.loads(line)  # a line written some other way
    return state["timestamp"], state["icao24"]


# The states of one exported file, and a clock that can move either way through them.
class Player:

    # Load every state as its JSON line, sorted by time; the clock starts at the first one.
    def __init__(self, path):
        rows = []
        with open_jsonl(path) as f:
            for line in f:
                timestamp, icao24 = _time_and_icao24(line)
                rows.append((timestamp, icao24, line.rstrip("\n") + "\n"))
        if not rows:
            raise ValueError(f"{path} has no states, try another time or a bigger area")
        rows.sort(key=lambda row: row[0])
        self.times = [row[0] for row in rows]
        self.icao24s = [row[1] for row in rows]
        self.lines = [row[2] for row in rows]
        self.first, self.last = self.times[0], self.times[-1]
        self.clock = self.first

    # Move the clock by seconds (negative goes back), staying inside the file.
    def skip(self, seconds):
        self.clock = min(max(self.clock + seconds, self.first), self.last)

    # JSON lines of the latest state of each aircraft heard in the GONE_AFTER_S before the clock.
    def snapshot(self):
        begin = bisect.bisect_left(self.times, self.clock - GONE_AFTER_S)
        end = bisect.bisect_right(self.times, self.clock)
        latest = {}
        for i in range(begin, end):
            latest[self.icao24s[i]] = self.lines[i]
        return list(latest.values())


# Replace out_path in one step, so the frontend never reads a half-written file.
def write_snapshot(lines, out_path):
    temporary = out_path + ".tmp"
    with open(temporary, "w") as out:
        out.writelines(lines)
    os.replace(temporary, out_path)


# Update the status, unless a newer start or stop has replaced this thread. Returns False then.
def _set(generation, **fields):
    with _lock:
        if generation != _generation:
            return False
        _status.update(fields)
        return True


# The background thread: download the day if needed, export it once, then write a
# snapshot every interval_s until stopped, moving the clock speed * interval_s each time.
# At the end of the file it keeps showing the last moment, so skip(-300) still works.
def _run(generation, date, start, hours, speed, interval_s, out_path, root):
    global _player
    try:
        if day_dir(date, root) is None:
            _set(generation, state="downloading")
            swap_day(date, root, on_progress=lambda done, total: _set(
                generation, done_bytes=done, total_bytes=total))
        export = os.path.join(root, "exports", f"{date}_{start.replace(':', '')}_{hours:g}h.jsonl.gz")
        if not os.path.exists(export):
            if not _set(generation, state="exporting", export_done=0, export_total=1):
                return
            with _export_lock:
                if not os.path.exists(export):  # another start may have just made it
                    def on_export_progress(done, total):
                        _set(generation, export_done=done, export_total=total)
                    export_day(date, export, start, hours, root=root, on_progress=on_export_progress)
        
        _set(generation, state="loading_player")
        player = Player(export)
        with _lock:
            if generation != _generation:
                return
            _player = player
            _status.update(state="playing", clock=player.clock, first=player.first, last=player.last)

        next_write = time.monotonic()
        while True:
            with _lock:
                if generation != _generation:
                    return
                write_snapshot(player.snapshot(), out_path)
                _status["clock"] = player.clock
                player.skip(speed * interval_s)
            next_write += interval_s
            time.sleep(max(0.0, next_write - time.monotonic()))
    except Exception as error:
        _set(generation, state="failed", error=str(error))


# Enter a date: show that day from start ("HH:MM" UTC) for hours in history.jsonl.
# Replaces whatever history was playing, and returns at once; follow it with history_status().
# A day that isn't on disk is downloaded first (~10 min), replacing the swap day.
def start_history(date, start="16:00", hours=1.0, speed=1.0, interval_s=1.0,
                  out_path=HISTORY_FILE, root=ARCHIVE_DIR):
    global _generation, _player
    with _lock:
        _generation += 1
        _player = None
        _status.update(state="loading", date=date, clock=None, first=None, last=None,
                       done_bytes=0, total_bytes=0, export_done=0, export_total=1, error=None)
        generation = _generation
        write_snapshot([], out_path)
    threading.Thread(target=_run, daemon=True,
                     args=(generation, date, start, hours, speed, interval_s, out_path, root)).start()


# Exit date mode: stop writing history.jsonl (the file stays). A download that already
# started can't be cancelled; it finishes in the background and the day stays on disk.
def stop_history():
    global _generation, _player
    with _lock:
        _generation += 1
        _player = None
        _status.update(state="stopped", clock=None, first=None, last=None)


# Go forward (seconds > 0) or back (seconds < 0) in the day that's playing.
# The frontend sees it on the next snapshot, within interval_s.
def skip(seconds):
    with _lock:
        if _player is None:
            raise RuntimeError("no history day is playing")
        _player.skip(seconds)
        _status["clock"] = _player.clock


# For the frontend to poll: {"state": "idle"|"loading"|"downloading"|"exporting"|"playing"|
# "stopped"|"failed", "date", "clock" (Unix s of the moment shown), "first", "last",
# "done_bytes", "total_bytes" (while downloading), "error"}
def history_status():
    with _lock:
        return dict(_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a history day into frontend/public/history.jsonl.")
    parser.add_argument("date", help="YYYY-MM-DD")
    parser.add_argument("--start", default="16:00", help="UTC start time, HH:MM (default 16:00)")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast as real time")
    args = parser.parse_args()
    start_history(args.date, args.start, args.hours, args.speed)
    try:
        while True:
            status = history_status()
            if status["state"] == "failed":
                raise SystemExit(status["error"])
            if status["clock"] is not None:
                when = datetime.datetime.fromtimestamp(status["clock"], datetime.timezone.utc)
                print(f"\r{when:%Y-%m-%d %H:%M:%S} UTC", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        stop_history()
        print("\nstopped")