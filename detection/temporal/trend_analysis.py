"""Numeric temporal analysis independent of the shared telemetry schema.

Callers select the telemetry field, units, time window, and domain thresholds.
These helpers describe numerical shape only and do not label a flight unsafe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def linear_slope(points: Sequence[tuple[float, float]]) -> float | None:
    """Return least-squares value-units per second, or None if underdetermined."""
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def is_monotonic(values: Sequence[float], *, increasing: bool, tolerance: float = 0.0) -> bool:
    """Check whether values trend in one direction within an absolute tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if len(values) < 2:
        return False
    pairs = zip(values, values[1:])
    if increasing:
        return all(right >= left - tolerance for left, right in pairs)
    return all(right <= left + tolerance for left, right in pairs)


def range_of(values: Sequence[float]) -> float | None:
    """Return max-minus-min, or None when there are no values."""
    if not values:
        return None
    return max(values) - min(values)


def slope_change(points: Sequence[tuple[float, float]]) -> float | None:
    """Estimate change in slope per second using adjacent segment slopes.

    The returned unit is value-units per second squared. At least three
    distinct timestamps are needed. Unevenly sampled data is supported.
    """
    if len(points) < 3:
        return None
    segment_slopes: list[tuple[float, float]] = []
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing")
        segment_slopes.append(((t0 + t1) / 2, (v1 - v0) / dt))
    return linear_slope(segment_slopes)


def persistence_ratio(
    values: Sequence[float], *, direction: str, tolerance: float = 0.0
) -> float | None:
    """Return the fraction of adjacent steps consistent with a direction.

    Flat steps within tolerance count as consistent. Returns None when fewer
    than two samples are available. Direction must be ``"increasing"`` or
    ``"decreasing"``.
    """
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if len(values) < 2:
        return None
    deltas = (right - left for left, right in zip(values, values[1:]))
    if direction == "increasing":
        consistent = sum(delta >= -tolerance for delta in deltas)
    else:
        consistent = sum(delta <= tolerance for delta in deltas)
    return consistent / (len(values) - 1)


def reversal_count(values: Sequence[float], *, tolerance: float = 0.0) -> int:
    """Count meaningful direction changes, ignoring changes within tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    prior_direction = 0
    reversals = 0
    for left, right in zip(values, values[1:]):
        delta = right - left
        direction = 1 if delta > tolerance else -1 if delta < -tolerance else 0
        if direction and prior_direction and direction != prior_direction:
            reversals += 1
        if direction:
            prior_direction = direction
    return reversals


@dataclass(frozen=True, slots=True)
class TrendSummary:
    """Unit-agnostic summary for a time-ordered numeric signal."""

    slope_per_second: float | None
    slope_change_per_second_squared: float | None
    persistence_increasing: float | None
    persistence_decreasing: float | None
    value_range: float | None
    reversals: int


def summarize_trend(
    points: Sequence[tuple[float, float]], *, tolerance: float = 0.0
) -> TrendSummary:
    """Summarize a numeric series without applying aviation thresholds."""
    times = [timestamp for timestamp, _ in points]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    values = [value for _, value in points]
    return TrendSummary(
        slope_per_second=linear_slope(points),
        slope_change_per_second_squared=slope_change(points),
        persistence_increasing=persistence_ratio(
            values, direction="increasing", tolerance=tolerance
        ),
        persistence_decreasing=persistence_ratio(
            values, direction="decreasing", tolerance=tolerance
        ),
        value_range=range_of(values),
        reversals=reversal_count(values, tolerance=tolerance),
    )
