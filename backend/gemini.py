"""Ask Gemini about one aircraft, for the map's aircraft popup.

Needs the google-genai package (pip install google-genai, inside .venv) and GEMINI_API_KEY,
either in a .env file at the repo root (GEMINI_API_KEY=...) or exported in the shell.
The key stays on the server; the browser only sends the aircraft's FlightState.
"""

import json
import os
import threading
import uuid

MODEL = "gemini-3.8-flash"
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
TIMEOUT_S = 60
MAX_JOBS = 100  # finished answers kept for the browser to collect; the oldest are dropped

_jobs = {}  # job id -> {"status": "working"} / {"status": "done", "answer"} / {"status": "error", "error"}
_jobs_lock = threading.Lock()

PROMPT = """Describe this aircraft for someone watching ADS-B traffic on a map. Reply in plain
text with no markdown, bullets or blank lines: exactly these 8 lines, in this order, each short.

1. "Live flight" if origin is "live", "Historical flight" if "history", "Simulated flight" if "sim".
2. "Callsign: " and flight_id, or "Callsign: none" if flight_id is just the icao24.
3. "Registration: " and aircraft.registration, or "Registration: unknown".
4. The size class and category, e.g. "Large A3 aircraft"; if type_code is known add the model,
   e.g. "Large A3 aircraft, Boeing 737-800 (B738)".
5. One sentence on the vertical situation, e.g. "Descending through 10,600 feet at about 1,400
   feet per minute toward a selected altitude of approximately 6,000 feet." Say climbing,
   descending, level at, or on the ground. Round altitudes to the nearest 100 feet and rates to
   the nearest 100 feet per minute. Leave out the selected altitude if it is null.
6. "Traveling at a ground speed of N knots", in whole knots.
7. "Squawking: " and the squawk code, or "Squawking: unknown".
8. One sentence on anything unusual: an emergency squawk (7500 hijack, 7600 radio failure, 7700
   emergency), an emergency status other than "none", a descent steeper than 3,000 feet per
   minute, faster than 250 knots below 10,000 feet, a stale position, or an anomaly other than
   "none". If there is nothing, write exactly: "There are no emergency codes, steep descents,
   or other anomalies reported in the data."

Use only the data below; missing values are null.

Aircraft (FlightState JSON):
{state}"""


# Put KEY=value lines from .env into the environment, without overriding anything already set.
def load_env_file(path=None):
    path = path or ENV_FILE
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            key, equals, value = line.strip().partition("=")
            if equals and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Gemini's short plain-text answer about one aircraft, given its FlightState.
# Raises RuntimeError with a readable message if the key or the package is missing.
def ask_about_aircraft(state):
    load_env_file()
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY isn't set: add it to .env, then ask again")
    try:
        from google import genai  # imported here so the rest of the server runs without it
    except ImportError:
        raise RuntimeError("google-genai isn't installed: run the server with .venv/bin/python "
                           "after .venv/bin/pip install google-genai")
    client = genai.Client()
    interaction = client.interactions.create(
        model=MODEL,
        input=PROMPT.format(state=json.dumps(state, indent=2)),
        timeout=TIMEOUT_S,
    )
    return interaction.output_text


# Ask about an aircraft in a background thread and return a job id at once; the browser then
# polls question_status(job_id). Holding one request open while Gemini thinks doesn't work:
# browsers drop connections that stay silent too long (Firefox does after any network change).
def start_question(state):
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "working"}
        while len(_jobs) > MAX_JOBS:
            del _jobs[next(iter(_jobs))]  # dicts keep insertion order, so this is the oldest

    def run():
        try:
            result = {"status": "done", "answer": ask_about_aircraft(state)}
        except Exception as error:  # no key, bad key, rate limit, network
            result = {"status": "error", "error": str(error)}
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id] = result

    threading.Thread(target=run, daemon=True).start()
    return job_id


# {"status": "working"}, {"status": "done", "answer": ...} or {"status": "error", "error": ...};
# None for a job id that doesn't exist (or was dropped as too old).
def question_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None
