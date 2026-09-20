from __future__ import annotations

from typing import Any

from .pipeline import IntegrationResult


class IntegratedEpisode:
    def __init__(self) -> None:
        self.results: list[IntegrationResult] = []

    def record(self, result: IntegrationResult) -> None:
        if not isinstance(result, IntegrationResult):
            raise TypeError("result must be IntegrationResult")
        self.results.append(result)

    @property
    def total_reward(self) -> float:
        return sum(result.total_reward for result in self.results)

    @property
    def steps(self) -> int:
        return sum(result.steps for result in self.results)

    @property
    def success(self) -> bool:
        return bool(self.results) and all(result.success for result in self.results)


def run_integrated_episode(
    pipeline: Any,
    episodes: int = 1,
    *args: Any,
    **kwargs: Any,
) -> IntegratedEpisode:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    episode = IntegratedEpisode()

    for _ in range(episodes):
        result = pipeline.run(*args, **kwargs)
        episode.record(result)

    return episode
