from __future__ import annotations
from math import isfinite

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecordedStep:
    observation: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]




# AGENTFORGE_CANONICAL_TRAJECTORY_SCHEMA_V2

@dataclass(frozen=True)
class TrajectoryStep:
    """Canonical benchmark trajectory step.

    A trajectory step is the complete observable transition record for one
    environment step.  It deliberately carries task/scenario identity and
    seed information so trajectories are reproducible and independently
    attributable during benchmark evaluation.
    """

    episode_id: str
    task_id: str
    scenario_id: str
    seed: int
    step_index: int
    observation: Any
    action: Any
    reward_total: float
    reward_components: Mapping[str, float] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty.")

        if not self.task_id:
            raise ValueError("task_id must not be empty.")

        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty.")

        if self.step_index < 0:
            raise ValueError("step_index must be non-negative.")

        if not isfinite(float(self.reward_total)):
            raise ValueError("reward_total must be finite.")

        duplicate_names = (
            len(self.reward_components)
            != len(set(self.reward_components.keys()))
        )

        if duplicate_names:
            raise ValueError("reward component names must be unique.")

        for name, value in self.reward_components.items():
            if not name:
                raise ValueError(
                    "reward component names must not be empty."
                )

            if not isfinite(float(value)):
                raise ValueError(
                    f"reward component {name!r} must be finite."
                )

        if self.terminated and self.truncated:
            raise ValueError(
                "terminated and truncated cannot both be True."
            )

    @property
    def reward(self) -> float:
        """Compatibility alias for legacy trajectory consumers."""
        return float(self.reward_total)

class TrajectoryRecorder:
    def __init__(self) -> None:
        self._steps: list[RecordedStep] = []
        self._closed = False

    def record(
        self,
        *,
        observation: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> RecordedStep:
        if self._closed:
            raise RuntimeError("trajectory recorder is closed")

        if terminated and truncated:
            raise ValueError(
                "terminated and truncated cannot both be true"
            )

        step = RecordedStep(
            observation=observation,
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )

        self._steps.append(step)
        return step

    def steps(self) -> tuple[RecordedStep, ...]:
        return tuple(self._steps)

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed
