"""Causal fixed-rate ADS-B sequence extraction for temporal research models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

STEPS = 31
INTERVAL_SECONDS = 10.0
HISTORY_SECONDS = 300.0
MAX_INTERPOLATION_GAP_SECONDS = 30.0

# Flight-state values and masks retain their original order; the final channels
# encode anchor-relative ground track geometry without exposing absolute location.
VALUE_NAMES = (
    "altitude_baro_ft", "ground_speed_kt", "vertical_rate_fpm",
    "track_sin", "track_cos", "on_ground",
)
MASK_NAMES = (
    "altitude_valid", "ground_speed_valid", "vertical_rate_valid",
    "track_valid", "on_ground_valid",
)
POSITION_VALUE_NAMES = ("east_position_nm_from_anchor", "north_position_nm_from_anchor")
POSITION_MASK_NAME = "position_valid"
CHANNEL_NAMES = VALUE_NAMES + MASK_NAMES + POSITION_VALUE_NAMES + (POSITION_MASK_NAME,)
VALUE_MASK_CHANNEL = {0: 6, 1: 7, 2: 8, 3: 9, 4: 9, 5: 10, 11: 13, 12: 13}
NORMALIZED_VALUE_CHANNELS = (0, 1, 2, 3, 4, 11, 12)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _samples(rows: Sequence[dict[str, Any]], field: str) -> list[tuple[float, float]]:
    result = []
    for row in rows:
        timestamp = _number(row.get("timestamp_unix_s"))
        value = _number(row.get(field))
        if timestamp is not None and value is not None:
            result.append((timestamp, value))
    return result


def _bracketed_value(samples: list[tuple[float, float]], target: float) -> float | None:
    """Linearly interpolate only between nearby observed values; never extrapolate."""
    if not samples:
        return None
    lo, hi = 0, len(samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid][0] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(samples) and abs(samples[lo][0] - target) < 1e-6:
        return samples[lo][1]
    if lo == 0 or lo == len(samples):
        return None
    left_t, left_v = samples[lo - 1]
    right_t, right_v = samples[lo]
    gap = right_t - left_t
    if gap <= 0 or gap > MAX_INTERPOLATION_GAP_SECONDS:
        return None
    fraction = (target - left_t) / gap
    return left_v + fraction * (right_v - left_v)


def _track_at(rows: Sequence[dict[str, Any]], target: float) -> float | None:
    samples = _samples(rows, "track_deg")
    if not samples:
        return None
    lo, hi = 0, len(samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid][0] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(samples) and abs(samples[lo][0] - target) < 1e-6:
        return samples[lo][1] % 360.0
    if lo == 0 or lo == len(samples):
        return None
    left_t, left_heading = samples[lo - 1]
    right_t, right_heading = samples[lo]
    gap = right_t - left_t
    if gap <= 0 or gap > MAX_INTERPOLATION_GAP_SECONDS:
        return None
    delta = (right_heading - left_heading + 180.0) % 360.0 - 180.0
    return (left_heading + delta * ((target - left_t) / gap)) % 360.0


def _longitude_at(rows: Sequence[dict[str, Any]], target: float) -> float | None:
    """Interpolate longitude along the short arc across the antimeridian."""
    samples = _samples(rows, "longitude_deg")
    if not samples:
        return None
    lo, hi = 0, len(samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid][0] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(samples) and abs(samples[lo][0] - target) < 1e-6:
        return ((samples[lo][1] + 180.0) % 360.0) - 180.0
    if lo == 0 or lo == len(samples):
        return None
    left_t, left_lon = samples[lo - 1]
    right_t, right_lon = samples[lo]
    gap = right_t - left_t
    if gap <= 0 or gap > MAX_INTERPOLATION_GAP_SECONDS:
        return None
    delta = (right_lon - left_lon + 180.0) % 360.0 - 180.0
    return ((left_lon + delta * ((target - left_t) / gap) + 180.0) % 360.0) - 180.0


def _ground_at(rows: Sequence[dict[str, Any]], target: float) -> bool | None:
    """Use the most recent categorical state only while it is at most 30s old."""
    selected: tuple[float, bool] | None = None
    for row in rows:
        timestamp = _number(row.get("timestamp_unix_s"))
        value = row.get("on_ground")
        if timestamp is None or not isinstance(value, bool) or timestamp > target:
            continue
        if selected is None or timestamp > selected[0]:
            selected = (timestamp, value)
    if selected is None or target - selected[0] > MAX_INTERPOLATION_GAP_SECONDS:
        return None
    return selected[1]


def resample_sequence(
    rows: Sequence[dict[str, Any]], anchor_unix_s: float,
) -> list[list[float]]:
    """Return causal 31-step observations through the anchor, inclusive."""
    anchor = _number(anchor_unix_s)
    if anchor is None:
        raise ValueError("Anchor must be finite")
    timestamps = [_number(row.get("timestamp_unix_s")) for row in rows]
    if any(value is None for value in timestamps):
        raise ValueError("Observation timestamps must be finite")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Observation timestamps must be strictly increasing")
    if timestamps and float(timestamps[-1]) > anchor + 1e-6:
        raise ValueError("Future observations are not allowed in causal sequences")

    altitude = _samples(rows, "altitude_baro_ft")
    speed = _samples(rows, "ground_speed_kt")
    vertical_rate = _samples(rows, "vertical_rate_fpm")
    latitude = _samples(rows, "latitude_deg")
    anchor_lat = _bracketed_value(latitude, anchor)
    anchor_lon = _longitude_at(rows, anchor)
    output: list[list[float]] = []
    for step in range(STEPS):
        target = anchor - HISTORY_SECONDS + step * INTERVAL_SECONDS
        alt = _bracketed_value(altitude, target)
        gs = _bracketed_value(speed, target)
        vr = _bracketed_value(vertical_rate, target)
        heading = _track_at(rows, target)
        ground = _ground_at(rows, target)
        sample_lat = _bracketed_value(latitude, target)
        sample_lon = _longitude_at(rows, target)
        position_valid = anchor_lat is not None and anchor_lon is not None and sample_lat is not None and sample_lon is not None
        if position_valid:
            # Local tangent-plane offsets are stable over a five-minute history,
            # shift-invariant, and wrap correctly at the antimeridian.
            delta_lon = (sample_lon - anchor_lon + 180.0) % 360.0 - 180.0
            east_nm = 60.0 * delta_lon * math.cos(math.radians(anchor_lat))
            north_nm = 60.0 * (sample_lat - anchor_lat)
        else:
            east_nm = north_nm = 0.0
        track_sin = math.sin(math.radians(heading)) if heading is not None else None
        track_cos = math.cos(math.radians(heading)) if heading is not None else None
        values = [alt, gs, vr, track_sin, track_cos,
                  float(ground) if ground is not None else None]
        masks = [float(value is not None) for value in values[:3]]
        masks.extend((float(heading is not None), float(ground is not None)))
        output.append(
            [float(value) if value is not None else 0.0 for value in values]
            + masks + [east_nm, north_nm, float(position_valid)]
        )
    return output
