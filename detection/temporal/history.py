"""Bounded, schema-neutral history primitives for temporal detectors."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Sample(Generic[T]):
    """A timestamped opaque sample; payload interpretation belongs to callers."""

    timestamp: float
    value: T


class TimeWindow(Generic[T]):
    """Keep recent samples within a duration and optional count limit."""

    def __init__(self, duration_seconds: float, max_samples: int | None = None) -> None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        self.duration_seconds = duration_seconds
        self.max_samples = max_samples
        self._samples: Deque[Sample[T]] = deque()

    def append(self, timestamp: float, value: T) -> None:
        """Append a chronological sample and evict expired/overflow samples."""
        if self._samples and timestamp < self._samples[-1].timestamp:
            raise ValueError("samples must be appended in timestamp order")
        self._samples.append(Sample(timestamp, value))
        cutoff = timestamp - self.duration_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
        if self.max_samples is not None:
            while len(self._samples) > self.max_samples:
                self._samples.popleft()

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample[T]]:
        return iter(tuple(self._samples))

    def snapshot(self) -> tuple[Sample[T], ...]:
        return tuple(self._samples)


class FlightHistory(Generic[T]):
    """Manage a separate bounded time window for each flight identifier."""

    def __init__(self, duration_seconds: float, max_samples_per_flight: int | None = None) -> None:
        self.duration_seconds = duration_seconds
        self.max_samples_per_flight = max_samples_per_flight
        self._windows: dict[str, TimeWindow[T]] = {}

    def append(self, flight_id: str, timestamp: float, value: T) -> None:
        window = self._windows.get(flight_id)
        if window is None:
            window = TimeWindow(self.duration_seconds, self.max_samples_per_flight)
            self._windows[flight_id] = window
        window.append(timestamp, value)

    def get(self, flight_id: str) -> tuple[Sample[T], ...]:
        window = self._windows.get(flight_id)
        return window.snapshot() if window is not None else ()

    def discard(self, flight_id: str) -> None:
        self._windows.pop(flight_id, None)

    def clear(self) -> None:
        self._windows.clear()

    def flight_ids(self) -> Iterable[str]:
        return self._windows.keys()
