from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .components import RewardComponent


@dataclass
class RewardTrace:
    """Ordered evaluator-side record of reward components."""

    components: list[RewardComponent] = field(default_factory=list)

    def add(self, component: RewardComponent) -> None:
        if not isinstance(component, RewardComponent):
            raise TypeError("component must be a RewardComponent")
        self.components.append(component)

    def extend(self, components: Iterable[RewardComponent]) -> None:
        for component in components:
            self.add(component)

    @property
    def total(self) -> float:
        return sum(component.value for component in self.components)

    def as_tuple(self) -> tuple[RewardComponent, ...]:
        return tuple(self.components)
