from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trajectory_recorder import TrajectoryRecorder


@dataclass(frozen=True)
class EpisodeResult:
    observation: Any
    total_reward: float
    step_count: int
    terminated: bool
    truncated: bool
    trajectory: tuple[Any, ...]


class EpisodeRunner:
    def __init__(self, environment: Any) -> None:
        self.environment = environment

    def run(
        self,
        actions: list[Any] | tuple[Any, ...],
    ) -> EpisodeResult:
        observation = self.environment.reset()

        if isinstance(observation, tuple):
            current_observation = observation[0]
        else:
            current_observation = observation

        recorder = TrajectoryRecorder()
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        observations = [current_observation]

        for action in actions:
            result = self.environment.step(action)

            if not isinstance(result, tuple) or len(result) != 5:
                raise ValueError(
                    "environment.step() must return "
                    "(observation, reward, terminated, truncated, info)"
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = result

            recorder.record(
                observation=current_observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

            total_reward += float(reward)
            step_count += 1
            current_observation = next_observation
            observations.append(current_observation)

            if terminated or truncated:
                break

        recorder.close()

        return EpisodeResult(
            observation=current_observation,
            total_reward=total_reward,
            step_count=step_count,
            terminated=terminated,
            truncated=truncated,
            trajectory=tuple(observations),
        )
