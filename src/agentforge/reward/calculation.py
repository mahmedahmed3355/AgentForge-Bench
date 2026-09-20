from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .components import RewardComponent


@dataclass(frozen=True)
class RewardCalculation:
    """Immutable calculation result for one reward evaluation."""

    total: float
    components: tuple[RewardComponent, ...] = ()

    def __post_init__(self) -> None:
        value = float(self.total)

        if not math.isfinite(value):
            raise ValueError("reward total must be finite")

    @classmethod
    def from_components(
        cls,
        components: Iterable[RewardComponent],
    ) -> "RewardCalculation":
        normalized = tuple(components)
        total = math.fsum(component.value for component in normalized)
        return cls(total=total, components=normalized)
