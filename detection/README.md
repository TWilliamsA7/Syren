# Detection Engine

Skeleton for the detection component. Implementation should begin after the shared `FlightState` and `DetectionResult` schemas are agreed.

## Current flow

1. Parse a protocol-shaped mapping with `flight_state_from_mapping` (or provide a `FlightState`).
2. Call `build_default_engine().update(state)` in timestamp order per aircraft.
3. Serialize the `DetectionResult` with `to_mapping` for the API/UI boundary.

The default rules cover rapid and accelerating descent, ground-speed decay, abrupt track changes, heading reversals, altitude oscillation, telemetry gaps, barometric altitude/rate disagreement, and squawk codes 7500/7600/7700. `update_fleet(states)` additionally screens projected aircraft conflicts using a short constant-velocity horizon. Thresholds and projections are initial heuristics and should be tuned against simulation and historical replay.

## Boundaries

- Detection consumes canonical shared models; it does not own or redefine them.
- Missing source measurements remain missing.
- Prototype risk scores must not be presented as calibrated safety probabilities.
- Keep the initial streaming path CPU-first; assess acceleration using fleet benchmarks.
- The protocol has no engine, hydraulic, electrical, or warning-system fields, so a system-failure rule cannot be implemented from current inputs. Add that detector only if the shared contract gains relevant telemetry.
- Fleet conflict screening requires a synchronized snapshot with position, track, speed, and altitude. It uses a local constant-velocity projection and is not a separation-assurance system.
