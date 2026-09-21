from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from agentforge.verifier import VerificationResult, VerifierExecutor

from .episode_runner import EpisodeResult, EpisodeRunner


@dataclass(frozen=True)
class EvaluationResult:
    episodes: tuple[EpisodeResult, ...]
    episode_count: int
    total_reward: float
    verification_results: tuple[VerificationResult, ...] = ()

    @property
    def mean_reward(self) -> float:
        if self.episode_count == 0:
            return 0.0
        return self.total_reward / self.episode_count


class EvaluationRunner:
    def __init__(
        self,
        environment_factory: Any,
        *,
        verifier_executor: VerifierExecutor | None = None,
        task_spec_resolver: Any | None = None,
        scenario_resolver: Any | None = None,
    ) -> None:
        self.environment_factory = environment_factory
        self.verifier_executor = verifier_executor
        self.task_spec_resolver = task_spec_resolver
        self.scenario_resolver = scenario_resolver

    def _run_single(
        self,
        agent: object,
        episode_index: int,
    ) -> EpisodeResult:
        raise NotImplementedError(
            "_run_single is not used by the legacy action-sequence "
            "evaluation API"
        )

    def run(
        self,
        task_id: str,
        action_sequences: Iterable[Iterable[Any]],
    ) -> EvaluationResult:
        results: list[EpisodeResult] = []
        verification_results: list[VerificationResult] = []

        for episode_index, actions in enumerate(action_sequences):
            environment = self.environment_factory.create(task_id)
            runner = EpisodeRunner(environment)

            try:
                result = runner.run(tuple(actions))
                results.append(result)

                if self.verifier_executor is not None:
                    if self.task_spec_resolver is None:
                        raise RuntimeError(
                            "Verifier integration requires task_spec_resolver"
                        )

                    task_spec = self.task_spec_resolver(task_id)

                    if self.scenario_resolver is None:
                        scenario = {
                            "task_id": task_id,
                            "episode_index": episode_index,
                        }
                    else:
                        scenario = self.scenario_resolver(
                            task_id,
                            episode_index,
                            result,
                        )

                    verification = self.verifier_executor.execute(
                        task_spec=task_spec,
                        scenario=scenario,
                        episode_result=result,
                        trajectory=result.trajectory,
                        metadata={
                            "task_id": task_id,
                            "episode_index": episode_index,
                        },
                    )

                    verification_results.append(verification)

            finally:
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
            verification_results=tuple(verification_results),
        )
