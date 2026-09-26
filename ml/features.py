"""Causal kinematic features shared by training and future streaming inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

FEATURE_NAMES = (
    "altitude_last_ft",
    "altitude_delta_ft",
    "altitude_range_ft",
    "altitude_available",
    "speed_last_kt",
    "speed_delta_kt",
    "speed_range_kt",
    "speed_available",
    "vertical_rate_last_fpm",
    "vertical_rate_mean_fpm",
    "vertical_rate_min_fpm",
    "vertical_rate_max_fpm",
    "vertical_rate_available",
    "turn_abs_total_deg",
    "turn_rate_max_deg_s",
    "track_available",
    "ground_fraction",
)

TEMPORAL_SCALES_SECONDS = (30, 60, 180, 300)
TEMPORAL_EXTRA_NAMES = tuple(
    f"{name}_{seconds}s"
    for seconds in TEMPORAL_SCALES_SECONDS
    for name in ("altitude_trend_ft_s", "speed_trend_kt_s", "vertical_rate_mean_fpm", "turn_rate_deg_s")
) + (
    "observation_count", "max_observation_gap_s", "altitude_valid_fraction",
    "speed_valid_fraction", "vertical_rate_valid_fraction", "track_valid_fraction",
    "airborne_at_anchor", "climbing_at_anchor", "descending_at_anchor",
)
TEMPORAL_FEATURE_NAMES = FEATURE_NAMES + TEMPORAL_EXTRA_NAMES


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _series(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    return [number for row in rows if (number := finite(row.get(field))) is not None]


def _kinematic_summary(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    return values[-1], values[-1] - values[0], max(values) - min(values), 1.0


def extract_features(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Extract fixed-size features from observations at or before the anchor.

    Status, squawk, emergency, identity, callsign, date, and coordinates are
    intentionally excluded: the target is a future emergency declaration.
    """
    if len(rows) < 2:
        raise ValueError("At least two observations are required")
    times = [finite(row.get("timestamp_unix_s")) for row in rows]
    if any(t is None for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Observation timestamps must be finite and strictly increasing")
    valid_times = [float(t) for t in times if t is not None]

    altitude = _series(rows, "altitude_baro_ft")
    speed = _series(rows, "ground_speed_kt")
    vertical_rate = _series(rows, "vertical_rate_fpm")
    alt_last, alt_delta, alt_range, alt_available = _kinematic_summary(altitude)
    speed_last, speed_delta, speed_range, speed_available = _kinematic_summary(speed)
    vr_last = vertical_rate[-1] if vertical_rate else 0.0
    vr_mean = sum(vertical_rate) / len(vertical_rate) if vertical_rate else 0.0
    vr_min = min(vertical_rate) if vertical_rate else 0.0
    vr_max = max(vertical_rate) if vertical_rate else 0.0

    heading_points = [
        (float(row["timestamp_unix_s"]), heading)
        for row in rows
        if (heading := finite(row.get("track_deg"))) is not None
    ]
    turns: list[float] = []
    turn_rates: list[float] = []
    for (previous_time, previous_heading), (now, heading) in zip(heading_points, heading_points[1:]):
        delta = abs((heading - previous_heading + 180.0) % 360.0 - 180.0)
        turns.append(delta)
        turn_rates.append(delta / (now - previous_time))

    observed_ground = [row["on_ground"] for row in rows if isinstance(row.get("on_ground"), bool)]
    return {
        "altitude_last_ft": alt_last,
        "altitude_delta_ft": alt_delta,
        "altitude_range_ft": alt_range,
        "altitude_available": alt_available,
        "speed_last_kt": speed_last,
        "speed_delta_kt": speed_delta,
        "speed_range_kt": speed_range,
        "speed_available": speed_available,
        "vertical_rate_last_fpm": vr_last,
        "vertical_rate_mean_fpm": vr_mean,
        "vertical_rate_min_fpm": vr_min,
        "vertical_rate_max_fpm": vr_max,
        "vertical_rate_available": float(bool(vertical_rate)),
        "turn_abs_total_deg": sum(turns),
        "turn_rate_max_deg_s": max(turn_rates) if turn_rates else 0.0,
        "track_available": float(bool(heading_points)),
        "ground_fraction": sum(observed_ground) / len(observed_ground) if observed_ground else 0.0,
    }


def extract_temporal_features(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Add multi-scale causal trends and observation quality to the base features."""
    features = extract_features(rows)
    anchor = float(rows[-1]["timestamp_unix_s"])
    for seconds in TEMPORAL_SCALES_SECONDS:
        window = [row for row in rows if float(row["timestamp_unix_s"]) >= anchor - seconds]
        altitude = [(float(row["timestamp_unix_s"]), value) for row in window
                    if (value := finite(row.get("altitude_baro_ft"))) is not None]
        speed = [(float(row["timestamp_unix_s"]), value) for row in window
                 if (value := finite(row.get("ground_speed_kt"))) is not None]
        rates = [value for row in window if (value := finite(row.get("vertical_rate_fpm"))) is not None]
        tracks = [(float(row["timestamp_unix_s"]), value) for row in window
                  if (value := finite(row.get("track_deg"))) is not None]

        def trend(series: list[tuple[float, float]]) -> float:
            return (series[-1][1] - series[0][1]) / (series[-1][0] - series[0][0]) if len(series) > 1 else 0.0

        total_turn = sum(abs((now - previous + 180.0) % 360.0 - 180.0)
                         for (_, previous), (_, now) in zip(tracks, tracks[1:]))
        features[f"altitude_trend_ft_s_{seconds}s"] = trend(altitude)
        features[f"speed_trend_kt_s_{seconds}s"] = trend(speed)
        features[f"vertical_rate_mean_fpm_{seconds}s"] = sum(rates) / len(rates) if rates else 0.0
        features[f"turn_rate_deg_s_{seconds}s"] = total_turn / (tracks[-1][0] - tracks[0][0]) if len(tracks) > 1 else 0.0

    times = [float(row["timestamp_unix_s"]) for row in rows]
    features["observation_count"] = float(len(rows))
    features["max_observation_gap_s"] = max(b - a for a, b in zip(times, times[1:]))
    for name, field in (
        ("altitude_valid_fraction", "altitude_baro_ft"),
        ("speed_valid_fraction", "ground_speed_kt"),
        ("vertical_rate_valid_fraction", "vertical_rate_fpm"),
        ("track_valid_fraction", "track_deg"),
    ):
        features[name] = sum(finite(row.get(field)) is not None for row in rows) / len(rows)
    last = rows[-1]
    last_rate = finite(last.get("vertical_rate_fpm"))
    features["airborne_at_anchor"] = float(last.get("on_ground") is False)
    features["climbing_at_anchor"] = float(last_rate is not None and last_rate > 300)
    features["descending_at_anchor"] = float(last_rate is not None and last_rate < -300)
    if tuple(features) != TEMPORAL_FEATURE_NAMES:
        raise ValueError("Temporal feature schema changed unexpectedly")
    return features
