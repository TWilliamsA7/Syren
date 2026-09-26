# Detection Engine

Skeleton for the detection component. Implementation should begin after the shared `FlightState` and `DetectionResult` schemas are agreed.

## Planned flow

1. Accept one canonical flight state through a stable update interface.
2. Run per-sample rule detectors.
3. Update per-flight temporal history and trend signals.
4. Aggregate signals into a prototype risk assessment.
5. Return the agreed detection result.

## Boundaries

- Detection consumes canonical shared models; it does not own or redefine them.
- Missing source measurements remain missing.
- Prototype risk scores must not be presented as calibrated safety probabilities.
- Keep the initial streaming path CPU-first; assess acceleration using fleet benchmarks.
