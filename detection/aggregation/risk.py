"""Schema-neutral risk aggregation extension point; policy is not selected yet."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class WeightedSignal(Protocol):
    """Minimum internal view an eventual aggregation policy may require."""

    @property
    def severity(self) -> float: ...

    @property
    def weight(self) -> float: ...


def combine_prototype_risk(signals: Iterable[WeightedSignal]) -> float | None:
    """TODO: choose and document aggregation semantics after signal schema agreement."""
    raise NotImplementedError("risk aggregation policy has not been agreed")
