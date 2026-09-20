from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .episode import EpisodeResult


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate result for a collection of episodes."""

    episodes: int
    successful_episodes: int
    success_rate: float
    mean_reward: float
    total_steps: int


def summarize(results: Iterable[EpisodeResult]) -> EvaluationSummary:
    items = list(results)

    if not items:
        return EvaluationSummary(
            episodes=0,
            successful_episodes=0,
            success_rate=0.0,
            mean_reward=0.0,
            total_steps=0,
        )

    successful = sum(1 for result in items if result.success)
    reward = sum(result.total_reward for result in items)
    steps = sum(result.steps for result in items)

    return EvaluationSummary(
        episodes=len(items),
        successful_episodes=successful,
        success_rate=successful / len(items),
        mean_reward=reward / len(items),
        total_steps=steps,
    )
