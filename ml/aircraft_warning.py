"""Causal, per-aircraft warning I/O for the frozen two-rule research pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from detection.models import FlightState, flight_state_from_mapping
from ml.features import extract_temporal_features

SPEED_LOSS_KT_PER_SECOND = 0.12069301307201385
ALTITUDE_REVERSAL_FT = 100.0
HISTORY_SECONDS = 300.0
MIN_HISTORY_SECONDS = 180.0
MIN_OBSERVATIONS = 10
MAX_GAP_SECONDS = 120.0
EVALUATION_INTERVAL_SECONDS = 60.0
ALERT_SUPPRESSION_SECONDS = 600.0
Mode = Literal["combined", "speed_loss", "altitude_reversal"]
Status = Literal["collecting_history", "clear", "warning", "suppressed"]


@dataclass(frozen=True, slots=True)
class WarningSignal:
    type: str
    value: float
    threshold: float
    window_seconds: float
    message: str

    def to_mapping(self) -> dict[str, str | float]:
        return {"type": self.type, "value": self.value, "threshold": self.threshold,
                "window_seconds": self.window_seconds, "message": self.message}


@dataclass(frozen=True, slots=True)
class PredictionResult:
    icao24: str
    flight_id: str
    timestamp: float
    status: Status
    evaluated: bool
    alert: bool
    signals: tuple[WarningSignal, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {"icao24": self.icao24, "flight_id": self.flight_id,
                "timestamp": self.timestamp, "status": self.status,
                "evaluated": self.evaluated, "alert": self.alert,
                "signals": [signal.to_mapping() for signal in self.signals]}


class AircraftWarningEngine:
    """Update one aircraft at a time; histories and cooldowns remain separate."""

    def __init__(self, mode: Mode = "combined") -> None:
        if mode not in ("combined", "speed_loss", "altitude_reversal"):
            raise ValueError(f"Unsupported warning mode: {mode}")
        self.mode = mode
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._last_seen: dict[str, float] = {}
        self._flight_id: dict[str, str] = {}
        self._last_evaluation: dict[str, float] = {}
        self._last_alert: dict[str, float] = {}

    def reset(self, icao24: str | None = None) -> None:
        if icao24 is None:
            self._history.clear()
            self._last_seen.clear()
            self._flight_id.clear()
            self._last_evaluation.clear()
            self._last_alert.clear()
            return
        key = icao24.strip().lower()
        for collection in (self._history, self._last_seen, self._flight_id,
                           self._last_evaluation, self._last_alert):
            collection.pop(key, None)

    def update(self, state: FlightState | Mapping[str, Any]) -> PredictionResult:
        if isinstance(state, Mapping):
            state = flight_state_from_mapping(state)
        if not isinstance(state, FlightState):
            raise TypeError("state must be FlightState or a protocol-shaped mapping")
        if isinstance(state.timestamp, bool) or not isinstance(state.timestamp, (int, float)) or not math.isfinite(state.timestamp):
            raise ValueError("timestamp must be finite Unix seconds")
        if not isinstance(state.icao24, str) or not state.icao24.strip():
            raise ValueError("icao24 must be a nonempty string")
        icao = state.icao24.strip().lower()
        flight_id = "".join(state.flight_id.split()) or icao
        now = float(state.timestamp)
        previous = self._last_seen.get(icao)
        if previous is not None and now < previous:
            raise ValueError(f"out-of-order state for {icao}")
        if (previous is not None and now - previous > MAX_GAP_SECONDS
                or icao in self._flight_id and self._flight_id[icao] != flight_id
                or state.position.on_ground is True):
            self.reset(icao)
        if state.position.on_ground is True:
            self._last_seen[icao] = now
            self._flight_id[icao] = flight_id
            return PredictionResult(icao, flight_id, now, "collecting_history", False, False)
        if state.position.stale is True:
            # A stale position must not enter a future trend window. Keep the
            # alert cooldown so a brief reception outage cannot cause a repeat.
            self._history.pop(icao, None)
            self._last_evaluation.pop(icao, None)
            self._last_seen[icao] = now
            self._flight_id[icao] = flight_id
            return PredictionResult(icao, flight_id, now, "collecting_history", False, False)
        if previous == now and icao in self._history:
            # One timestamp represents one observation. A corrected duplicate
            # replaces it but cannot produce a second alert at the same time.
            history = self._history[icao]
            history.pop()
            history.append(self._observation(state))
            return PredictionResult(icao, flight_id, now, "clear", False, False)
        self._last_seen[icao] = now
        self._flight_id[icao] = flight_id
        history = self._history.setdefault(icao, deque())
        history.append(self._observation(state))
        while history and history[0]["timestamp_unix_s"] < now - HISTORY_SECONDS:
            history.popleft()
        if (len(history) < MIN_OBSERVATIONS
                or now - history[0]["timestamp_unix_s"] < MIN_HISTORY_SECONDS):
            return PredictionResult(icao, flight_id, now, "collecting_history", False, False)
        last_evaluation = self._last_evaluation.get(icao)
        if last_evaluation is not None and now - last_evaluation < EVALUATION_INTERVAL_SECONDS:
            return PredictionResult(icao, flight_id, now, "clear", False, False)
        self._last_evaluation[icao] = now
        features = extract_temporal_features(tuple(history))
        signals = self._signals(features)
        if not signals:
            return PredictionResult(icao, flight_id, now, "clear", True, False)
        last_alert = self._last_alert.get(icao, -math.inf)
        if now - last_alert < ALERT_SUPPRESSION_SECONDS:
            return PredictionResult(icao, flight_id, now, "suppressed", True, False, signals)
        self._last_alert[icao] = now
        return PredictionResult(icao, flight_id, now, "warning", True, True, signals)

    @staticmethod
    def _observation(state: FlightState) -> dict[str, Any]:
        return {
            "timestamp_unix_s": float(state.timestamp),
            "altitude_baro_ft": state.position.altitude_baro_ft,
            "ground_speed_kt": state.kinematics.ground_speed_kts,
            "vertical_rate_fpm": state.kinematics.vertical_rate_baro_fpm,
            "track_deg": state.kinematics.track_deg,
            "on_ground": state.position.on_ground,
        }

    def _signals(self, features: dict[str, float]) -> tuple[WarningSignal, ...]:
        if features["airborne_at_anchor"] < 0.5:
            return ()
        signals = []
        if self.mode in ("combined", "speed_loss") and features["speed_valid_fraction"] >= 0.5:
            loss = -features["speed_trend_kt_s_60s"]
            if loss >= SPEED_LOSS_KT_PER_SECOND:
                signals.append(WarningSignal(
                    "SPEED_LOSS", loss, SPEED_LOSS_KT_PER_SECOND, 60.0,
                    f"Ground speed fell about {loss * 60:.1f} kt/min over the last minute"))
        if self.mode in ("combined", "altitude_reversal") and features["altitude_valid_fraction"] >= 0.5:
            reversal = features["altitude_range_ft"] - abs(features["altitude_delta_ft"])
            if reversal >= ALTITUDE_REVERSAL_FT:
                signals.append(WarningSignal(
                    "ALTITUDE_REVERSAL", reversal, ALTITUDE_REVERSAL_FT, 300.0,
                    f"Five-minute altitude reversal measure {reversal:.0f} ft"))
        return tuple(signals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("combined", "speed_loss", "altitude_reversal"), default="combined")
    parser.add_argument("--input-jsonl", default="-", help="protocol FlightState JSONL, or '-' for stdin")
    args = parser.parse_args()
    engine = AircraftWarningEngine(args.mode)
    source = sys.stdin if args.input_jsonl == "-" else Path(args.input_jsonl).open(encoding="utf-8")
    try:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                result = engine.update(json.loads(line))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid aircraft state at line {line_number}: {exc}") from exc
            print(json.dumps(result.to_mapping(), separators=(",", ":"), allow_nan=False))
    finally:
        if source is not sys.stdin:
            source.close()


if __name__ == "__main__":
    main()
