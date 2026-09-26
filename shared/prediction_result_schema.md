# Per-aircraft predictive warning I/O

`AircraftWarningEngine.update` accepts one chronological `FlightState` or the
protocol mapping in `docs/protocol.md` and returns one `PredictionResult`.
Keep one engine instance for an interleaved stream of aircraft; histories and
ten-minute alert suppression are maintained separately by ICAO24. Call
`reset(icao24)` before a new replay or after an externally known flight-leg
boundary.

For a direct JSONL input/output stream, run:

```sh
python -m ml.aircraft_warning --input-jsonl flight_states.jsonl
```

It writes one prediction JSON object for every input state. Interleaved
aircraft IDs are supported; each aircraft's timestamps must be chronological.

```json
{
  "icao24": "39348e",
  "flight_id": "TVF39ZD",
  "timestamp": 1763189550.6,
  "status": "warning",
  "evaluated": true,
  "alert": true,
  "signals": [
    {
      "type": "SPEED_LOSS",
      "value": 0.22463768193562086,
      "threshold": 0.12069301307201385,
      "window_seconds": 60.0,
      "message": "Ground speed fell about 13.5 kt/min over the last minute"
    }
  ]
}
```

`status` is `collecting_history`, `clear`, `warning`, or `suppressed`.
`evaluated` is false during history warmup or between one-minute scoring
intervals. `alert` is true only when a new, unsuppressed warning is emitted.
`signals` lists measured behaviors whenever scoring found an active rule,
including a suppressed warning. `SPEED_LOSS` values are knots per second;
`ALTITUDE_REVERSAL` values are feet. These values are not probabilities.

The frozen research thresholds are speed loss of at least
0.12069301307201385 knots/second over 60 seconds and altitude reversal of at
least 100 feet over five minutes. Both require airborne state and at least
half the respective measurements present. Predictions use only prior and
current kinematics; squawk and emergency status are ignored.

The research replay counts a declaration as preceded by a warning when an
unsuppressed alert occurs within the prior 20 minutes. It separately reports
alerts more than 10 minutes early, 2–10 minutes early, and under 2 minutes
early. The predictor itself does not receive declaration labels or compute
lead time.
