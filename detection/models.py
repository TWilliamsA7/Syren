"""Typed detection-side models matching the contracts in docs/protocol.md."""

from __future__ import annotations

import math
import re
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
    accuracy_m: float | None = None
    stale: bool | None = None


@dataclass(frozen=True, slots=True)
class Kinematics:
    ground_speed_kts: float | None = None
    track_deg: float | None = None
    vertical_rate_baro_fpm: float | None = None
    vertical_rate_geom_fpm: float | None = None


@dataclass(frozen=True, slots=True)
class Navigation:
    selected_altitude_ft: float | None = None


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
    nav: Navigation = field(default_factory=Navigation)
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
    def optional_number(section_value: Mapping[str, Any], key: str) -> float | None:
        value = section_value.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number or null")
        return float(value)

    def optional_string(section_value: Mapping[str, Any], key: str) -> str | None:
        value = section_value.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string or null")
        return value

    def optional_boolean(section_value: Mapping[str, Any], key: str) -> bool | None:
        value = section_value.get(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean or null")
        return value

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
    nav = section("nav")
    status = section("status")
    icao24 = data.get("icao24")
    if not isinstance(icao24, str) or not icao24.strip():
        raise ValueError("icao24 must be a non-empty string")
    icao24 = icao24.strip().lower()
    timestamp = data.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite numeric Unix seconds")
    flight_id_value = data.get("flight_id")
    if flight_id_value is not None and not isinstance(flight_id_value, str):
        raise ValueError("flight_id must be a string or null")
    flight_id = re.sub(r"\s+", "", flight_id_value or "")
    if not flight_id:
        flight_id = icao24

    for key in ("registration", "type_code", "category"):
        optional_string(aircraft, key)
    for key in ("latitude", "longitude", "altitude_baro_ft", "altitude_geom_ft", "accuracy_m"):
        optional_number(position, key)
    for key in ("ground_speed_kts", "track_deg", "vertical_rate_baro_fpm", "vertical_rate_geom_fpm"):
        optional_number(kinematics, key)
    optional_number(nav, "selected_altitude_ft")
    for key in ("squawk", "emergency"):
        optional_string(status, key)
    optional_number(status, "seen_age_s")
    optional_string(position, "source")
    optional_string(data, "origin")
    optional_boolean(position, "on_ground")
    optional_boolean(position, "stale")
    latitude = optional_number(position, "latitude")
    longitude = optional_number(position, "longitude")
    accuracy = optional_number(position, "accuracy_m")
    if latitude is not None and not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if longitude is not None and not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be between -180 and 180 degrees")
    if accuracy is not None and accuracy < 0:
        raise ValueError("accuracy_m must be non-negative")

    return FlightState(
        timestamp=float(timestamp),
        icao24=icao24,
        flight_id=flight_id,
        aircraft=Aircraft(**{key: aircraft.get(key) for key in Aircraft.__dataclass_fields__}),
        position=Position(**{
            "latitude": optional_number(position, "latitude"),
            "longitude": optional_number(position, "longitude"),
            "altitude_baro_ft": optional_number(position, "altitude_baro_ft"),
            "altitude_geom_ft": optional_number(position, "altitude_geom_ft"),
            "on_ground": optional_boolean(position, "on_ground"),
            "source": optional_string(position, "source"),
            "accuracy_m": optional_number(position, "accuracy_m"),
            "stale": optional_boolean(position, "stale"),
        }),
        kinematics=Kinematics(**{key: optional_number(kinematics, key) for key in Kinematics.__dataclass_fields__}),
        nav=Navigation(selected_altitude_ft=optional_number(nav, "selected_altitude_ft")),
        status=Status(
            squawk=optional_string(status, "squawk"),
            emergency=optional_string(status, "emergency"),
            seen_age_s=optional_number(status, "seen_age_s"),
        ),
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
