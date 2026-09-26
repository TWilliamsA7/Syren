# Detection Engine V1

Rule-based V1 implementation for the protocol in `docs/protocol.md`.

## Use

Create one engine per simulation or replay session. Call `update` for a single state or `update_fleet` once for each synchronized fleet snapshot. Results are `DetectionResult` records; call `to_mapping()` at the JSON/API boundary. Input must be chronological per aircraft. Call `reset(icao24)` before replaying an aircraft from an earlier time, or `reset()` between complete runs.

`DetectionEngine()` and `build_default_engine()` create the default V1 rules. A custom detector sequence may be injected for a smaller configuration or later model-backed detector.

The default rules cover rapid and accelerating descent, ground-speed decay, abrupt track changes, heading reversals, altitude oscillation, telemetry gaps, altitude/rate disagreement (barometric or geometric), and squawk codes 7500/7600/7700. Fleet snapshots additionally screen projected aircraft conflicts using a short constant-velocity horizon. Thresholds and projections are initial heuristics and should be tuned against simulation and historical replay.

## Boundaries

- Detection consumes canonical shared models; it does not own or redefine them.
- Missing source measurements remain missing.
- Prototype risk scores must not be presented as calibrated safety probabilities.
- Keep the initial streaming path CPU-first; assess acceleration using fleet benchmarks.
- The protocol has no engine, hydraulic, electrical, or warning-system fields. V1 flags observable flight-behavior anomalies; a future learned model can infer risk patterns from suitable historical examples without direct fault flags.
- Fleet conflict screening requires a synchronized snapshot with position, track, speed, and altitude. It uses a local constant-velocity projection and is not a separation-assurance system.
- `nav.selected_altitude_ft` and `position.accuracy_m` are parsed and preserved. V1 does not alert on selected-altitude deviation because the protocol has no autopilot mode or clearance context; accuracy is used to widen the projected-conflict threshold conservatively.
