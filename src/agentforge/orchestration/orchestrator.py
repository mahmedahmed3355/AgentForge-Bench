from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .context import OrchestrationContext
from .episode import EpisodeOrchestrator, EpisodeResult
from .evaluation import EvaluationSummary, summarize


@dataclass(frozen=True)
class OrchestrationResult:
    """Complete result of one orchestration request."""

    context: OrchestrationContext
    episodes: tuple[EpisodeResult, ...]
    summary: EvaluationSummary


class Orchestrator:
    """Top-level framework coordinator.

    Dependencies are injected as callables. No concrete task implementation
    is imported here, preventing accidental coupling to protected benchmark
    tasks.
    """

    def __init__(
        self,
        episode_factory: Callable[[], EpisodeOrchestrator],
    ) -> None:
        self._episode_factory = episode_factory

    def evaluate(
        self,
        *,
        context: OrchestrationContext,
        episodes: int = 1,
    ) -> OrchestrationResult:
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        results: list[EpisodeResult] = []

        for _ in range(episodes):
            runner = self._episode_factory()
            results.append(
                runner.run(
                    task_id=context.task_id,
                    scenario_id=context.scenario_id,
                )
            )

        frozen = tuple(results)

        return OrchestrationResult(
            context=context,
            episodes=frozen,
            summary=summarize(frozen),
        )
