import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend.gemini import question_status, start_question
from data.history import available_range, days_on_disk
from data.replay import history_status, skip, start_history, stop_history

PORT = 8000
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_FILE = os.path.join(REPO, "frontend", "public", "data.jsonl")
_LIVE_CACHE_LOCK = threading.Lock()
_live_cache_fingerprint = object()
_live_cache_body = b"[]"
_live_cache_etag = '"empty"'


def _live_feed_payload():
    """Read and encode the live feed only when the adapter replaces it."""
    global _live_cache_fingerprint, _live_cache_body, _live_cache_etag
    try:
        stat = os.stat(LIVE_FILE)
        fingerprint = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        fingerprint = None

    with _LIVE_CACHE_LOCK:
        if fingerprint != _live_cache_fingerprint:
            if fingerprint is None:
                aircraft = []
                _live_cache_etag = '"missing"'
            else:
                with open(LIVE_FILE, "r", encoding="utf-8") as feed:
                    aircraft = [json.loads(line) for line in feed if line.strip()]
                _live_cache_etag = f'"{fingerprint[0]:x}-{fingerprint[1]:x}"'
            _live_cache_body = json.dumps(aircraft, separators=(",", ":")).encode()
            _live_cache_fingerprint = fingerprint
        age = max(0.0, time.time() - stat.st_mtime) if fingerprint else None
        return _live_cache_body, _live_cache_etag, age


class Handler(BaseHTTPRequestHandler):

    # Send body back as JSON. If the browser already hung up there's no one to answer.
    def _reply(self, body, status=200):
        data = json.dumps(body).encode()
        self._reply_bytes(data, status)

    def _reply_bytes(self, data, status=200, etag=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if etag is not None:
                self.send_header("Cache-Control", "no-cache")
                self.send_header("ETag", etag)
                age = getattr(self, "_live_feed_age", None)
                self.send_header("X-Feed-Age-S", "unknown" if age is None else f"{age:.1f}")
            if status != 304:
                self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # The JSON body of a POST, or {} if there is none.
    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    # GET /api/history_status, polled every second; GET /api/history_days for the date picker;
    # GET /api/ask_aircraft/<job>, polled every second for a Gemini answer.
    def do_GET(self):
        if self.path.startswith("/api/ask_aircraft/"):
            status = question_status(self.path.rsplit("/", 1)[1].split("?", 1)[0])
            self._reply(status if status else {"error": "unknown question"}, 200 if status else 404)
        elif self.path.split("?", 1)[0] == "/api/aircraft":
            body, etag, age = _live_feed_payload()
            self._live_feed_age = age
            if self.headers.get("If-None-Match") == etag:
                self._reply_bytes(b"", 304, etag)
            else:
                self._reply_bytes(body, 200, etag)
        elif self.path == "/api/history_status":
            self._reply(history_status())
        elif self.path == "/api/history_days":
            first, last = available_range()
            self._reply({"first": first, "last": last, **days_on_disk()})
        else:
            self._reply({"error": "not found"}, 404)

    # POST /api/start_history {date, start}, /api/stop_history, /api/skip {seconds}:
    # replies with the new history status. POST /api/ask_aircraft {aircraft: FlightState}:
    # starts asking Gemini and replies {"job": id} at once; poll GET /api/ask_aircraft/<id>.
    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/ask_aircraft":
                return self._reply({"job": start_question(body["aircraft"])}, 202)
            elif self.path == "/api/start_history":
                start_history(body["date"], body.get("start", "16:00"))
            elif self.path == "/api/stop_history":
                stop_history()
            elif self.path == "/api/skip":
                skip(float(body["seconds"]))
            else:
                return self._reply({"error": "not found"}, 404)
        except RuntimeError as error:  # skip while nothing is playing
            return self._reply({"error": str(error)}, 409)
        except (KeyError, ValueError) as error:  # a missing field or bad JSON
            return self._reply({"error": f"bad request: {error!r}"}, 400)
        self._reply(history_status())

    # Don't print a line for every request; the status is polled every second.
    def log_message(self, format, *args):
        pass


# Start the live feed in its own process, then serve the history API until Ctrl-C.
if __name__ == "__main__":
    live = subprocess.Popen([sys.executable, "-m", "data.live_adapter", LIVE_FILE], cwd=REPO)
    print(f"history API on http://localhost:{PORT}")
    try:
        ThreadingHTTPServer(("", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.terminate()
