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

Create an aircraft feed for the frontend by attaching each result to its
original `FlightState`. Keep the input chronological per aircraft, and choose
`frontend/public/data.jsonl` as the output path when using the frontend's default
feed URL:

```sh
python -m detection.enrich_jsonl path/to/flight_states.jsonl frontend/public/data.jsonl
```

The resulting JSONL retains all flight-state fields and adds a `detection`
object containing `risk_score`, `severity`, and `anomalies`. Any non-empty
`detection.anomalies` list colors the aircraft red in the map. Emergency
transponder states remain red as well.

For plane-to-plane interaction checks, provide one JSON array of synchronized states per line:

```sh
python -m detection.harness --fleet-jsonl path/to/fleet_snapshots.jsonl
```

Measure the per-aircraft streaming path at several sizes:

```sh
python -m detection.harness --benchmark 100 1000 10000
```

Score a generated dataset against the detector types that its telemetry can
support:

```sh
python -m simulation.generate data/generated/train_001 --flights 1000 --seed 1
python -m detection.evaluate_generated data/generated/train_001
```

The evaluator requires the matching detector type for rapid descent, erratic
altitude, speed loss, low-altitude overspeed, squawk, and signal loss. It reports
engine failure and hijack as symptom alerts rather than causal classifications.
Route deviation is marked unsupported and excluded from scored denominators:
`FlightState` provides the observed path but no intended route. Normal-state
and anomaly-free-flight alert rates are reported separately. The evaluator uses
the single-state path, so it does not assess aircraft conflicts.

That benchmark intentionally excludes the pairwise conflict scan. The conflict rule is exercised by the smoke suite; benchmark it separately at realistic snapshot sizes before scaling it up, since this first implementation checks aircraft pairs directly.

The default rules cover rapid and accelerating descent, ground-speed decay,
sustained speed above 300 kt below 10,000 ft, abrupt track changes, heading
reversals, altitude oscillation, telemetry gaps, altitude/rate disagreement
(barometric or geometric), and squawk codes 7500/7600/7700. Expected descent
context suppresses ordinary speed-decay and accelerating-descent alerts during
planned descent and approach. A speed drop below 280 kt above 10,000 ft can
still alert during descent; rapid descent and near-ground overspeed remain
independent.
Fleet snapshots additionally screen projected aircraft conflicts using a short
constant-velocity horizon. Thresholds and projections are initial heuristics
and should be tuned against simulation and historical replay.

## Boundaries

- Detection consumes canonical shared models; it does not own or redefine them.
- Missing source measurements remain missing.
- Prototype risk scores must not be presented as calibrated safety probabilities.
- Keep the initial streaming path CPU-first; assess acceleration using fleet benchmarks.
- The protocol has no engine, hydraulic, electrical, or warning-system fields. V1 flags observable flight-behavior anomalies; a future learned model can infer risk patterns from suitable historical examples without direct fault flags.
- Fleet conflict screening requires a synchronized snapshot with position, track, speed, and altitude. It uses a local constant-velocity projection and is not a separation-assurance system.
- `nav.selected_altitude_ft` and `position.accuracy_m` are parsed and preserved. V1 does not alert on selected-altitude deviation because the protocol has no autopilot mode or clearance context; accuracy is used to widen the projected-conflict threshold conservatively.
