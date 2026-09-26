"""Typed detection-side models matching the contracts in docs/protocol.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Literal


Severity = Literal["normal", "advisory", "warning", "critical"]


@dataclass(frozen=True, slots=True)
class Aircraft:
    registration: str | None = None
    type_code: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class Position:
    latitude: float | None = None
    longitude: float | None = None
    altitude_baro_ft: float | None = None
    altitude_geom_ft: float | None = None
    on_ground: bool | None = None
    source: str | None = None
    stale: bool | None = None


@dataclass(frozen=True, slots=True)
class Kinematics:
    ground_speed_kts: float | None = None
    track_deg: float | None = None
    vertical_rate_baro_fpm: float | None = None
    vertical_rate_geom_fpm: float | None = None
    ias_kts: float | None = None
    roll_deg: float | None = None


@dataclass(frozen=True, slots=True)
class Status:
    squawk: str | None = None
    emergency: str | None = None
    seen_age_s: float | None = None


@dataclass(frozen=True, slots=True)
class FlightState:
    timestamp: float
    icao24: str
    flight_id: str
    aircraft: Aircraft = field(default_factory=Aircraft)
    position: Position = field(default_factory=Position)
    kinematics: Kinematics = field(default_factory=Kinematics)
    status: Status = field(default_factory=Status)
    origin: str | None = None


@dataclass(frozen=True, slots=True)
class Anomaly:
    type: str
    severity: float
    message: str


@dataclass(frozen=True, slots=True)
class DetectionResult:
    icao24: str
    flight_id: str
    timestamp: float
    risk_score: float
    severity: Severity
    anomalies: tuple[Anomaly, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        """Convert to the protocol's JSON-compatible DetectionResult shape."""
        return {
            "icao24": self.icao24,
            "flight_id": self.flight_id,
            "timestamp": self.timestamp,
            "risk_score": self.risk_score,
            "severity": self.severity,
            "anomalies": [
                {"type": anomaly.type, "severity": anomaly.severity, "message": anomaly.message}
                for anomaly in self.anomalies
            ],
        }


def flight_state_from_mapping(data: Mapping[str, Any]) -> FlightState:
    """Parse a protocol-shaped mapping while leaving omitted values as None."""
    def section(name: str) -> Mapping[str, Any]:
        value = data.get(name)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object when provided")
        return value

    aircraft = section("aircraft")
    position = section("position")
    kinematics = section("kinematics")
    status = section("status")
    icao24 = data.get("icao24")
    if not isinstance(icao24, str) or not icao24.strip():
        raise ValueError("icao24 must be a non-empty string")
    timestamp = data.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp must be numeric Unix seconds")
    flight_id = data.get("flight_id")
    if not isinstance(flight_id, str) or not flight_id.strip():
        flight_id = icao24

    return FlightState(
        timestamp=float(timestamp),
        icao24=icao24,
        flight_id=flight_id,
        aircraft=Aircraft(**{key: aircraft.get(key) for key in Aircraft.__dataclass_fields__}),
        position=Position(**{key: position.get(key) for key in Position.__dataclass_fields__}),
        kinematics=Kinematics(**{key: kinematics.get(key) for key in Kinematics.__dataclass_fields__}),
        status=Status(**{key: status.get(key) for key in Status.__dataclass_fields__}),
        origin=data.get("origin"),
    )


def severity_for_risk(risk: float) -> Severity:
    """Apply the thresholds specified by the shared protocol."""
    if not 0.0 <= risk <= 1.0:
        raise ValueError("risk must be in [0, 1]")
    if risk < 0.30:
        return "normal"
    if risk < 0.60:
        return "advisory"
    if risk < 0.85:
        return "warning"
    return "critical"
