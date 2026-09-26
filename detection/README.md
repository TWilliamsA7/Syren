# Detection Engine V1

Rule-based V1 implementation for the protocol in `docs/protocol.md`.

## Use

Create one engine per simulation or replay session. Call `update` for a single state or `update_fleet` once for each synchronized fleet snapshot. Results are `DetectionResult` records; call `to_mapping()` at the JSON/API boundary. Input must be chronological per aircraft. Call `reset(icao24)` before replaying an aircraft from an earlier time, or `reset()` between complete runs.

`DetectionEngine()` and `build_default_engine()` create the default V1 rules. A custom detector sequence may be injected for a smaller configuration or later model-backed detector.

## Quick verification on the Jetson

From the repository root, run the generated deterministic suite:

```sh
python -m detection.harness
```

It checks a clean normal baseline, every single-aircraft detector, and a projected two-aircraft conflict. It exits non-zero if an expected anomaly is missing or the baseline produces an alert. Run the regression tests with:

```sh
python -m unittest discover -s detection/tests -v
```

Replay one protocol `FlightState` JSON object per line and emit one result JSON object per line:

```sh
python -m detection.harness --input-jsonl path/to/flight_states.jsonl
```

For plane-to-plane interaction checks, provide one JSON array of synchronized states per line:

```sh
python -m detection.harness --fleet-jsonl path/to/fleet_snapshots.jsonl
```

Measure the per-aircraft streaming path at several sizes:

```sh
python -m detection.harness --benchmark 100 1000 10000
```

That benchmark intentionally excludes the pairwise conflict scan. The conflict rule is exercised by the smoke suite; benchmark it separately at realistic snapshot sizes before scaling it up, since this first implementation checks aircraft pairs directly.

The default rules cover rapid and accelerating descent, ground-speed decay, abrupt track changes, heading reversals, altitude oscillation, telemetry gaps, altitude/rate disagreement (barometric or geometric), and squawk codes 7500/7600/7700. Fleet snapshots additionally screen projected aircraft conflicts using a short constant-velocity horizon. Thresholds and projections are initial heuristics and should be tuned against simulation and historical replay.

## Boundaries

- Detection consumes canonical shared models; it does not own or redefine them.
- Missing source measurements remain missing.
- Prototype risk scores must not be presented as calibrated safety probabilities.
- Keep the initial streaming path CPU-first; assess acceleration using fleet benchmarks.
- The protocol has no engine, hydraulic, electrical, or warning-system fields. V1 flags observable flight-behavior anomalies; a future learned model can infer risk patterns from suitable historical examples without direct fault flags.
- Fleet conflict screening requires a synchronized snapshot with position, track, speed, and altitude. It uses a local constant-velocity projection and is not a separation-assurance system.
- `nav.selected_altitude_ft` and `position.accuracy_m` are parsed and preserved. V1 does not alert on selected-altitude deviation because the protocol has no autopilot mode or clearance context; accuracy is used to widen the projected-conflict threshold conservatively.
