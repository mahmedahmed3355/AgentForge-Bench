from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    reward: float
    steps: int
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not isinstance(self.reward, (int, float)):
            raise TypeError("reward must be numeric")
