"""Schema-neutral extension points for the future detection engine.

The shared FlightState and DetectionResult contracts are intentionally not
assumed here. Once agreed, an adapter/engine can connect them to these pieces.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar


StateT = TypeVar("StateT", contravariant=True)
SignalT = TypeVar("SignalT", covariant=True)


class Detector(Protocol[StateT, SignalT]):
    """A detector consumes one state and yields zero or more internal signals."""

    def update(self, state: StateT) -> tuple[SignalT, ...]:
        """Evaluate a state, updating any detector-owned temporal state."""
        ...


class DetectionEngine(Generic[StateT, SignalT]):
    """Placeholder orchestration over detectors, without a public result schema."""

    def __init__(self, detectors: tuple[Detector[StateT, SignalT], ...] = ()) -> None:
        self._detectors = detectors

    def collect_signals(self, state: StateT) -> tuple[SignalT, ...]:
        """Collect raw signals; aggregation and result conversion remain pending."""
        return tuple(signal for detector in self._detectors for signal in detector.update(state))
