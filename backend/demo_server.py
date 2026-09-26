"""Run the anomaly showcase continuously and serve detected aircraft to the UI.

From the repository root:
    python -m backend.demo_server --speed 20

The Vite development server proxies ``/api`` requests to this process.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from detection.engine import build_default_engine
from simulation.run import load_scenario, run_scenario


DEFAULT_SCENARIO = Path(__file__).resolve().parents[1] / "simulation" / "scenarios" / "anomaly_showcase.json"


class DemoState:
    def __init__(self, scenario: dict[str, Any], speed: float) -> None:
        self.scenario = scenario
        self.speed = speed
        self.aircraft: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.error: str | None = None
        self.running = True

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.aircraft.values())

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "scenario": self.scenario["name"],
                "speed": self.speed,
                "aircraft_count": len(self.aircraft),
                "error": self.error,
            }


def simulate(state: DemoState) -> None:
    """Continuously replay the selected scenario and publish detected states."""
    while state.running:
        engine = build_default_engine()
        with state.lock:
            state.aircraft.clear()
            state.error = None
        wall_start = time.monotonic()
        start_time = float(state.scenario["start_time"])
        try:
            for flight_state in run_scenario(state.scenario):
                if not state.running:
                    return
                effective_time = float(flight_state["timestamp"])
                effective_time += float(flight_state.get("status", {}).get("seen_age_s") or 0)
                elapsed = max(0.0, (effective_time - start_time) / state.speed)
                delay = wall_start + elapsed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

                result = engine.update_mapping(flight_state)
                enriched = {**flight_state, "detection": result}
                with state.lock:
                    state.aircraft[flight_state["icao24"]] = enriched
                    # Drop tracks that stopped reporting more than a minute ago.
                    for icao24, previous in tuple(state.aircraft.items()):
                        last_seen = float(previous["timestamp"])
                        last_seen += float(previous.get("status", {}).get("seen_age_s") or 0)
                        if effective_time - last_seen > 60:
                            del state.aircraft[icao24]
        except Exception as error:
            with state.lock:
                state.error = f"{type(error).__name__}: {error}"
                state.running = False
            return


def make_handler(state: DemoState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/aircraft":
                payload: Any = state.snapshot()
            elif path == "/api/status":
                payload = state.status()
            else:
                self.send_error(404)
                return
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO, help="scenario JSON (default: anomaly showcase)")
    parser.add_argument("--speed", type=float, default=20.0, help="simulation seconds per wall-clock second")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args(argv)
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")

    scenario = load_scenario(args.scenario)
    state = DemoState(scenario, args.speed)
    worker = threading.Thread(target=simulate, args=(state,), name="syren-simulation", daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Syren demo API: http://{args.host}:{args.port}/api/aircraft")
    print(f"Scenario: {scenario['name']} at {args.speed:g}x; press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        server.server_close()
        worker.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
