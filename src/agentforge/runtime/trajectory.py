"""Canonical trajectory schema for AgentForge-RL-Bench.

A trajectory records the complete interaction between an agent and a task
environment.  The schema is deliberately independent from any particular
environment implementation.

Canonical step fields:

    episode_id
    task_id
    scenario_id
    seed
    step_index
    observation
    action
    reward_total
    reward_components
    terminated
    truncated
    info

Task-specific metadata belongs in ``info`` rather than becoming mandatory
schema fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

from agentforge.runtime.contracts.rewards import RewardBreakdown


@dataclass(frozen=True)
class TrajectoryStep:
    """One canonical environment transition."""

    episode_id: str
    task_id: str
    scenario_id: str
    seed: int
    step_index: int
    observation: Any
    action: Any
    reward_total: float
    reward_components: RewardBreakdown
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty.")

        if not self.task_id:
            raise ValueError("task_id must not be empty.")

        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty.")

        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")

        if self.step_index < 0:
            raise ValueError("step_index must be non-negative.")

        if not isfinite(float(self.reward_total)):
            raise ValueError("reward_total must be finite.")

        if not isinstance(self.reward_components, RewardBreakdown):
            raise TypeError(
                "reward_components must be a RewardBreakdown."
            )

        if bool(self.terminated) and bool(self.truncated):
            raise ValueError(
                "terminated and truncated cannot both be true."
            )

        object.__setattr__(
            self,
            "info",
            dict(self.info),
        )


@dataclass(frozen=True)
class Trajectory:
    """Canonical ordered episode trajectory."""

    episode_id: str
    task_id: str
    scenario_id: str
    seed: int
    steps: tuple[TrajectoryStep, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty.")

        if not self.task_id:
            raise ValueError("task_id must not be empty.")

        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty.")

        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")

        previous = -1

        for step in self.steps:
            if step.episode_id != self.episode_id:
                raise ValueError(
                    "All trajectory steps must use the same episode_id."
                )

            if step.task_id != self.task_id:
                raise ValueError(
                    "All trajectory steps must use the same task_id."
                )

            if step.scenario_id != self.scenario_id:
                raise ValueError(
                    "All trajectory steps must use the same scenario_id."
                )

            if step.seed != self.seed:
                raise ValueError(
                    "All trajectory steps must use the same seed."
                )

            if step.step_index <= previous:
                raise ValueError(
                    "Trajectory step_index values must be strictly increasing."
                )

            previous = step.step_index

    @property
    def length(self) -> int:
        """Number of recorded environment transitions."""
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        """Total scalar reward across all recorded steps."""
        return sum(step.reward_total for step in self.steps)

    def append(self, step: TrajectoryStep) -> "Trajectory":
        """Return a new trajectory containing one additional step."""
        if self.steps and step.step_index <= self.steps[-1].step_index:
            raise ValueError(
                "New trajectory step must have a greater step_index."
            )

        return Trajectory(
            episode_id=self.episode_id,
            task_id=self.task_id,
            scenario_id=self.scenario_id,
            seed=self.seed,
            steps=self.steps + (step,),
        )
