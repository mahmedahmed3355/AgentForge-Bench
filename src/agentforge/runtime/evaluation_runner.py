from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .episode_runner import EpisodeResult, EpisodeRunner


@dataclass(frozen=True)
class EvaluationResult:
    episodes: tuple[EpisodeResult, ...]
    episode_count: int
    total_reward: float

    @property
    def mean_reward(self) -> float:
        if self.episode_count == 0:
            return 0.0
        return self.total_reward / self.episode_count


class EvaluationRunner:
    def __init__(self, environment_factory: Any) -> None:
        self.environment_factory = environment_factory

    def run(
        self,
        task_id: str,
        action_sequences: Iterable[Iterable[Any]],
    ) -> EvaluationResult:
        results: list[EpisodeResult] = []

        for actions in action_sequences:
            environment = self.environment_factory.create(task_id)
            runner = EpisodeRunner(environment)
            result = runner.run(tuple(actions))
            results.append(result)

            close = getattr(environment, "close", None)
            if callable(close):
                close()

        total_reward = sum(
            result.total_reward
            for result in results
        )

        return EvaluationResult(
            episodes=tuple(results),
            episode_count=len(results),
            total_reward=total_reward,
        )
