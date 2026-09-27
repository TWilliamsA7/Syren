# Syren

## Overview

Syren is a real-time aviation anomaly detection engine. It takes ADS-B flight telemetry
(position, altitude, ground speed, heading, vertical rate, squawk code and emergency status)
from live traffic and from archived days, converts every report into one common format (the
FlightState in `docs/protocol.md`), and runs each aircraft through a detection engine that
flags potential airborne emergencies as they develop.

### Core capabilities

- **Live monitoring:** US airspace from [adsb.lol](https://adsb.lol), refreshed every few
  seconds, run through the detection engine and shown on a map.
- **Anomaly detection:** rule-based detectors for emergency squawks (7500 hijack, 7600 radio
  failure, 7700 emergency), rapid and accelerating descents, altitude, speed and heading
  anomalies, heading oscillation, fast flight near the ground, conflicting traffic, and
  unreliable telemetry. See `detection/README.md`.
- **Historical replay:** any day since 2023-02-16 from the adsb.lol
  [globe_history](https://github.com/adsblol/globe_history_2026) archive can be replayed in the
  same view, with the same detection, and skipped forward or back.
- **Simulation:** synthetic flights with injected anomalies (engine failure, hijack, rapid
  descent, route deviation, signal loss and more) test the detectors against known ground truth.
- **Backtesting research:** `ml/` labels emergencies in archived days and accident traces to
  measure how early an emergency can be predicted. See `ml/README.md`.
- **Ask Gemini:** a short, structured plain-language summary of any aircraft on the map.

## Requirements

- Python 3.13 
- Node.js 20.19+ or 22.12+ with npm
- Internet access (live feed, map tiles, fonts, Gemini, history downloads)
- Optional: a Gemini API key, for "Ask Gemini"
- Optional: about 3 GB of disk per history day you download

Run everything from the repository root unless a step says otherwise.

## 1. Python environment

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` includes PyTorch and scikit-learn for the `ml/` research pipeline, which is
a large download. To run only the app (server, live feed, replay, detection, Gemini),
`pip install google-genai` is enough; everything else uses the standard library.

Always run the project with the venv's Python, `.venv/bin/python`, which has the packages
installed; the system `python3` doesn't. On Windows, use `.\.venv\Scripts\python.exe`
wherever this README says `.venv/bin/python`.

## 2. Gemini API key (optional)

`.env` is not committed. Create it in the repository root with one line:

```
GEMINI_API_KEY=your-key-here
```

Get a key at https://aistudio.google.com/apikey. The key stays on the Python server and is
never sent to the browser. Without it everything works except "Ask Gemini", which shows an
error.

## 3. Frontend dependencies

```sh
cd frontend
npm install
```

Install inside `frontend/`, not in the repository root.

## 4. History days (optional, for replay)

Replay reads days from `data/archive/` (not committed). Download the days listed in
`FIXED_DAYS` in `data/history.py` (each takes about 10 minutes and 3 GB of disk):

```sh
.venv/bin/python -m data.history setup
.venv/bin/python -m data.history list
```

Any other day from 2023-02-16 to yesterday can be picked in the app. It is downloaded then
(about 10 minutes) and replaces the previous non-fixed day.

## Run

Terminal 1, from the repository root:

```sh
.venv/bin/python -m backend.server
```

Terminal 2:

```sh
cd frontend
npm run dev
```

Open http://localhost:5173. Vite forwards `/api` requests to port 8000. Stop both with Ctrl-C.

### Using the app

- **Live:** planes are green; orange when a detector flags them (listed under
  "Detected anomalies"); yellow when the experimental prediction model expects trouble
  (listed under "Experimental behavior warnings"); red when squawking an emergency (listed
  under "Airspace").
- **Find aircraft:** enter a callsign, ICAO hex or registration to zoom to it and follow it.
- **Click a plane** to pin its details, then "Ask Gemini about this aircraft" for a summary.
- **Replay a day:** pick a date and a 24-hour UTC start time, then "Load replay". The first
  replay of a date and time builds an export in `data/archive/exports/`, which takes 1–3
  minutes for a whole-US hour; later replays of that date and time start in seconds. Use
  −5 min / +5 min to skip, and "Back to live" to return.



## Layout

| Folder | What it holds |
|---|---|
| `backend/` | API server (live feed, history, replay, Gemini) and the demo server |
| `data/` | Live adapter, history download, trace export, replay |
| `detection/` | Rule-based anomaly detection engine |
| `simulation/` | Synthetic flights and anomaly scenarios |
| `ml/` | Emergency-prediction research pipeline |
| `scripts/` | Tools that fetch archived days and accident traces for `ml/` |
| `shared/` | FlightState schema and validation |
| `frontend/` | React + deck.gl map |
| `docs/protocol.md` | FlightState and detection-result formats |
