from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class EpisodeResult:
    """Framework-neutral result produced by one orchestrated episode."""

    task_id: str
    scenario_id: str
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    success: bool
    trajectory: tuple[Any, ...] = field(default_factory=tuple)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class EpisodeOrchestrator:
    """Small deterministic orchestration primitive.

    The orchestration layer intentionally accepts callables instead of
    assuming a concrete environment or agent implementation. This keeps
    framework layers decoupled while giving higher-level runners one stable
    execution contract.
    """

    def __init__(
        self,
        *,
        reset: Callable[[], Any],
        step: Callable[[Any], Any],
        is_success: Callable[[Any], bool] | None = None,
        max_steps: int = 1000,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self._reset = reset
        self._step = step
        self._is_success = is_success or (lambda _: False)
        self._max_steps = max_steps

    def run(
        self,
        *,
        task_id: str,
        scenario_id: str,
    ) -> EpisodeResult:
        observation = self._reset()
        trajectory: list[Any] = [observation]
        total_reward = 0.0
        terminated = False
        truncated = False

        for step_index in range(1, self._max_steps + 1):
            result = self._step(observation)

            if not isinstance(result, tuple) or len(result) != 5:
                raise TypeError(
                    "step callable must return "
                    "(observation, reward, terminated, truncated, info)"
                )

            observation, reward, terminated, truncated, info = result
            total_reward += float(reward)
            trajectory.append(
                {
                    "step": step_index,
                    "observation": observation,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }
            )

            if terminated or truncated:
                break

        else:
            truncated = True

        success = bool(self._is_success(observation))

        return EpisodeResult(
            task_id=task_id,
            scenario_id=scenario_id,
            steps=len(trajectory) - 1,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            trajectory=tuple(trajectory),
        )
