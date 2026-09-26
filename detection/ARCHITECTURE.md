# Detection Component Contract

## Boundary

The detector is source-independent. Simulation, historical replay, and future live adapters must convert their observations into the `FlightState` shape in `docs/protocol.md`. Detection must not branch on `origin`; that field is for debugging only.

```text
FlightState or protocol mapping
        │
        ├── update(state) ──► per-aircraft rules/history ──► DetectionResult
        │
        └── update_fleet(snapshot) ──► per-aircraft rules/history
                                      + aircraft-pair screening
                                      ──► DetectionResult per aircraft
```

`DetectionEngine` owns orchestration, timestamp ordering, risk aggregation, and conversion to result records. Detectors own only their rule and any bounded per-aircraft history. The API or replay caller owns source ingestion, session lifetime, and delivery of result mappings to downstream components.

## Input and output

- `update` accepts a typed `FlightState` or a mapping in the protocol shape. It returns one result for the input timestamp.
- `update_mapping` is the JSON-ready single-state convenience method.
- `update_fleet` accepts one snapshot with at most one state per ICAO24. Call it with the latest synchronized state for each aircraft when pairwise conflicts are required.
- `DetectionResult.to_mapping()` emits the protocol result shape. Anomaly types are constrained by the `AnomalyType` registry.
- Samples must be chronological per ICAO24. Equal timestamps are accepted; an earlier timestamp is rejected. Use `reset(icao24)` before replaying that aircraft from an earlier point, or `reset()` between sessions.
- Missing telemetry remains missing. No detector uses `origin` as a feature.

## V1 stages

1. Parse and validate protocol data.
2. Run the registered per-aircraft rule and temporal detectors.
3. For fleet snapshots, screen projected aircraft pairs.
4. Aggregate anomaly severities into the prototype risk score and apply protocol risk labels.
5. Return the result for downstream serialization and streaming.

The risk combiner and detector thresholds are prototype heuristics, not calibrated probabilities. The conflict projection is a short-horizon constant-velocity screen, not a separation-assurance service.

## Extension points

- A new source belongs in a source adapter and must emit `FlightState`; it should not add source-specific logic to detector rules.
- A new rule implements the detector `update(state) -> Anomaly | None` contract and registers an `AnomalyType`.
- A later learned detector can implement the same single-state rule interface, keeping inference behind a detector adapter. Training, feature windows, and model artifacts belong in a separate model component.
- Fleet-level methods belong behind a fleet detector invoked by `update_fleet`; they should not make single-state calls depend on other aircraft.
- New anomaly types must be added to the protocol/UI vocabulary and the `AnomalyType` registry together.

## Outside this component

The backend/API/WebSocket, simulation and historical data adapters, UI, Jetson deployment automation, and trained model are separate components. This directory provides the detection boundary and harness to plug into those components; it does not claim the full application path is already connected.
