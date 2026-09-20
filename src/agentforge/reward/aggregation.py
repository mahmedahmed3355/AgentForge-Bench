from __future__ import annotations

import math
from typing import Iterable

from .calculation import RewardCalculation
from .components import RewardComponent


class RewardAggregator:
    """Aggregates reward components into a finite scalar reward."""

    def __init__(self) -> None:
        self._components: list[RewardComponent] = []

    def add(self, component: RewardComponent) -> None:
        if not isinstance(component, RewardComponent):
            raise TypeError("component must be a RewardComponent")
        self._components.append(component)

    def extend(self, components: Iterable[RewardComponent]) -> None:
        for component in components:
            self.add(component)

    def calculate(self) -> RewardCalculation:
        return RewardCalculation.from_components(self._components)

    def total(self) -> float:
        value = self.calculate().total
        if not math.isfinite(value):
            raise ValueError("aggregated reward must be finite")
        return value

    @property
    def components(self) -> tuple[RewardComponent, ...]:
        return tuple(self._components)
