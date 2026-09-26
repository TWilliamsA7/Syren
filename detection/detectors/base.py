"""Shared helpers for schema-backed rule detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from detection.models import Anomaly, FlightState


class RuleDetector(ABC):
    @abstractmethod
    def update(self, state: FlightState) -> Anomaly | None:
        """Return a current anomaly, or None when no rule is triggered."""
        raise NotImplementedError

    def reset(self, icao24: str | None = None) -> None:
        """Clear temporal state; stateless rules need no reset work."""
        return None


def bounded_severity(value: float) -> float:
    return max(0.0, min(1.0, value))
