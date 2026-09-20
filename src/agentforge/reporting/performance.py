from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceReport:
    episodes: int
    total_reward: float
    successful_episodes: int

    @property
    def mean_reward(self) -> float:
        if self.episodes < 0:
            raise ValueError("episodes must be non-negative")
        if self.successful_episodes < 0:
            raise ValueError("successful_episodes must be non-negative")
        if self.successful_episodes > self.episodes:
            raise ValueError(
                "successful_episodes cannot exceed episodes"
            )
        if self.episodes == 0:
            return 0.0
        return self.total_reward / self.episodes

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successful_episodes / self.episodes
