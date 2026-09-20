from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrajectoryAnalysis:
    rewards: Sequence[float]

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def steps(self) -> int:
        return len(self.rewards)
