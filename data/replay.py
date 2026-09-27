import argparse
from array import array
import bisect
import copy
import datetime
import json
import os
import re
import threading
import time

from data.history import ARCHIVE_DIR, day_dir, swap_day
from data.trace import export_day, open_jsonl
from detection.engine import build_default_engine
from ml.aircraft_warning import AircraftWarningEngine

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
_seek_revision = 0
_seek_target = None
_status = {"state": "idle", "date": None, "clock": None, "first": None, "last": None,
           "done_bytes": 0, "total_bytes": 0, "export_done": 0, "export_total": 0,
           "error": None, "seeking": False}


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
        source_stat = os.stat(path)
        is_compressed = path.endswith(".gz")
        plain_path = (f"{path}.{source_stat.st_size:x}-{source_stat.st_mtime_ns:x}.plain"
                      if is_compressed else path)
        if is_compressed and not os.path.exists(plain_path):
            temporary_path = plain_path + ".tmp"
            try:
                with open_jsonl(path) as source, open(temporary_path, "wb") as plain:
                    for line in source:
                        plain.write(line.encode("utf-8"))
                os.replace(temporary_path, plain_path)
            finally:
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass

        self.times = array("d")
        self.offsets = array("Q")
        ordered = True
        with open(plain_path, "rb") as f:
            for line in f:
                timestamp, _ = _time_and_icao24(line.decode("utf-8"))
                self.times.append(timestamp)
                self.offsets.append(f.tell() - len(line))
                if len(self.times) > 1 and timestamp < self.times[-2]:
                    ordered = False
        if not self.times:
            raise ValueError(f"{path} has no states, try another time or a bigger area")
        if not ordered:
            order = sorted(range(len(self.times)), key=self.times.__getitem__)
            self.times = array("d", (self.times[i] for i in order))
            self.offsets = array("Q", (self.offsets[i] for i in order))
        self._plain_path = plain_path
        self.first, self.last = self.times[0], self.times[-1]
        self.clock = self.first
        # Replay feeds are raw FlightStates too. Advance both engines in
        # timestamp order as the replay clock moves, and reset/replay on rewind.
        self._detection_engine = build_default_engine()
        self._warning_engine = AircraftWarningEngine()
        self._next_detection_index = 0
        self._detection_results = {}
        self._prediction_results = {}
        self._checkpoints = []
        self._next_checkpoint = self.first + 60.0
        self._save_checkpoint(self.first, 0)

    def _save_checkpoint(self, clock, cursor):
        self._checkpoints.append((clock, cursor, copy.deepcopy(self._detection_engine),
                                  copy.deepcopy(self._warning_engine),
                                  self._detection_results.copy(), self._prediction_results.copy()))

    def _restore_checkpoint(self, target):
        checkpoint = self._checkpoints[0]
        for candidate in self._checkpoints:
            if candidate[0] <= target and candidate[0] >= checkpoint[0]:
                checkpoint = candidate
        (self.clock, self._next_detection_index, self._detection_engine,
         self._warning_engine, self._detection_results, self._prediction_results) = copy.deepcopy(checkpoint)
        self._next_checkpoint = self.clock + 60.0

    def _advance_engines(self, target, cancelled=lambda: False, force_checkpoint=False):
        if force_checkpoint or target < self.clock:
            self._restore_checkpoint(target)
        end = bisect.bisect_right(self.times, target)
        with open(self._plain_path, "rb") as feed:
            while self._next_detection_index < end:
                # Checking the shared control lock for every record made large seeks
                # contend with status and skip requests. A batch keeps cancellation
                # responsive without putting a lock acquisition in the hot loop.
                if self._next_detection_index % 256 == 0 and cancelled():
                    return False
                index = self._next_detection_index
                offset = self.offsets[index]
                if feed.tell() != offset:
                    feed.seek(offset)
                state = json.loads(feed.readline())
                result = self._detection_engine.update_mapping(state)
                prediction = self._warning_engine.update(state).to_mapping()
                icao24 = state["icao24"].strip().lower()
                self._detection_results[icao24] = result
                self._prediction_results[icao24] = prediction
                self._next_detection_index += 1
                if self.times[index] >= self._next_checkpoint:
                    self._save_checkpoint(self.times[index], self._next_detection_index)
                    self._next_checkpoint = self.times[index] + 60.0
        self.clock = target
        return True

    # Move the clock by seconds (negative goes back), staying inside the file.
    def skip(self, seconds):
        self.clock = min(max(self.clock + seconds, self.first), self.last)

    # JSON lines of the latest state of each aircraft heard in the GONE_AFTER_S before the clock.
    def snapshot(self, target=None, cancelled=lambda: False, force_checkpoint=False):
        target = self.clock if target is None else min(max(target, self.first), self.last)
        if not self._advance_engines(target, cancelled, force_checkpoint):
            return None
        begin = bisect.bisect_left(self.times, target - GONE_AFTER_S)
        end = bisect.bisect_right(self.times, target)
        latest = {}
        with open(self._plain_path, "rb") as feed:
            for i in range(begin, end):
                offset = self.offsets[i]
                if feed.tell() != offset:
                    feed.seek(offset)
                line = feed.readline()
                state = json.loads(line)
                latest[state["icao24"]] = line
        enriched = []
        for icao24, line in latest.items():
            state = json.loads(line)
            detection = self._detection_results.get(icao24)
            if detection is not None:
                state["detection"] = detection
            prediction = self._prediction_results.get(icao24)
            if prediction is not None:
                state["prediction"] = prediction
            enriched.append(json.dumps(state, separators=(",", ":")) + "\n")
        return enriched


# Replace out_path in one step, so the frontend never reads a half-written file.
# On Windows a reader can briefly hold the destination without delete sharing.
# Retry that transient lock; if it persists for this tick, keep the last good
# snapshot and let the playback loop try again on its next update.
def write_snapshot(lines, out_path):
    temporary = out_path + ".tmp"
    with open(temporary, "w") as out:
        out.writelines(lines)
    for attempt in range(5):
        try:
            os.replace(temporary, out_path)
            return True
        except PermissionError:
            if attempt == 4:
                try:
                    os.remove(temporary)
                except FileNotFoundError:
                    pass
                return False
            time.sleep(0.05 * (2 ** attempt))


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
    global _player, _seek_target
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
            _seek_target = player.clock
            _status.update(state="playing", clock=player.clock, first=player.first, last=player.last)

        next_write = time.monotonic()
        handled_revision = -1
        while True:
            with _lock:
                if generation != _generation:
                    return
                revision = _seek_revision
                seeking = revision != handled_revision
                target = _seek_target if seeking else min(player.clock + speed * interval_s, player.last)
                _status["state"] = "seeking" if seeking and handled_revision >= 0 else "playing"
                _status["seeking"] = bool(seeking and handled_revision >= 0)

            def cancelled():
                with _lock:
                    return generation != _generation or revision != _seek_revision

            lines = player.snapshot(target, cancelled, force_checkpoint=seeking and handled_revision >= 0)
            if lines is None:
                continue
            temporary = out_path + f".{generation}.{revision}.tmp"
            with open(temporary, "w") as out:
                out.writelines(lines)
            with _lock:
                if generation != _generation:
                    try:
                        os.remove(temporary)
                    except FileNotFoundError:
                        pass
                    return
                if revision == _seek_revision and os.path.exists(temporary):
                    try:
                        os.replace(temporary, out_path)
                        _status["clock"] = target
                        _status["state"] = "playing"
                        _status["seeking"] = False
                        handled_revision = revision
                        _seek_target = target
                    except PermissionError:
                        try:
                            os.remove(temporary)
                        except FileNotFoundError:
                            pass
                elif os.path.exists(temporary):
                    try:
                        os.remove(temporary)
                    except FileNotFoundError:
                        pass
            if cancelled():
                continue
            next_write += interval_s
            time.sleep(max(0.0, next_write - time.monotonic()))
    except Exception as error:
        _set(generation, state="failed", error=str(error), seeking=False)


# Enter a date: show that day from start ("HH:MM" UTC) for hours in history.jsonl.
# Replaces whatever history was playing, and returns at once; follow it with history_status().
# A day that isn't on disk is downloaded first (~10 min), replacing the swap day.
def start_history(date, start="16:00", hours=1.0, speed=1.0, interval_s=1.0,
                  out_path=HISTORY_FILE, root=ARCHIVE_DIR):
    global _generation, _player, _seek_revision, _seek_target
    with _lock:
        _generation += 1
        _seek_revision = 0
        _seek_target = None
        _player = None
        _status.update(state="loading", date=date, clock=None, first=None, last=None,
                       done_bytes=0, total_bytes=0, export_done=0, export_total=1,
                       error=None, seeking=False)
        generation = _generation
        write_snapshot([], out_path)
    threading.Thread(target=_run, daemon=True,
                     args=(generation, date, start, hours, speed, interval_s, out_path, root)).start()


# Exit date mode: stop writing history.jsonl (the file stays). A download that already
# started can't be cancelled; it finishes in the background and the day stays on disk.
def stop_history():
    global _generation, _player, _seek_target, _seek_revision
    with _lock:
        _generation += 1
        _seek_revision += 1
        _seek_target = None
        _player = None
        _status.update(state="stopped", clock=None, first=None, last=None)
        _status["seeking"] = False


# Go forward (seconds > 0) or back (seconds < 0) in the day that's playing.
# The request returns immediately; the replay worker publishes its newest target.
def skip(seconds):
    global _seek_revision, _seek_target
    with _lock:
        if _player is None:
            raise RuntimeError("no history day is playing")
        base = _seek_target if _seek_target is not None else _status["clock"]
        _seek_target = min(max(base + seconds, _player.first), _player.last)
        _seek_revision += 1
        _status["clock"] = _seek_target
        _status["state"] = "seeking"
        _status["seeking"] = True


# For the frontend to poll: {"state": "idle"|"loading"|"downloading"|"exporting"|"playing"|
# "stopped"|"failed"|"seeking", "date", "clock" (Unix s of the moment shown), "first", "last",
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
