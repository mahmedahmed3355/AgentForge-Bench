from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .episode import IntegratedEpisode, run_integrated_episode


@dataclass(frozen=True)
class IntegratedEvaluation:
    episodes: int
    successful_episodes: int
    total_steps: int
    total_reward: float

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successful_episodes / self.episodes


def evaluate_integrated(
    pipeline: Any,
    episodes: int = 1,
    *args: Any,
    **kwargs: Any,
) -> IntegratedEvaluation:
    episode: IntegratedEpisode = run_integrated_episode(
        pipeline,
        episodes,
        *args,
        **kwargs,
    )

    successful = sum(
        1 for result in episode.results if result.success
    )

    return IntegratedEvaluation(
        episodes=episodes,
        successful_episodes=successful,
        total_steps=episode.steps,
        total_reward=episode.total_reward,
    )
