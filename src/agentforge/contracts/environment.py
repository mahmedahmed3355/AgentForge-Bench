from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Any, TypeAlias

import gymnasium as gym


Observation: TypeAlias = Any
Action: TypeAlias = Any
Info: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    """
    Canonical AgentForge environment step result.

    The result follows the Gymnasium five-value step contract:

        observation
        reward
        terminated
        truncated
        info
    """

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Info

    def __post_init__(self) -> None:
        if not isinstance(self.reward, (int, float)):
            raise TypeError("reward must be a scalar number")

        if not isfinite(float(self.reward)):
            raise ValueError("reward must be finite")

        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be bool")

        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be bool")

        if self.terminated and self.truncated:
            raise ValueError(
                "terminated and truncated cannot both be True"
            )

        if not isinstance(self.info, dict):
            raise TypeError("info must be a dict")


class EnvironmentContract(ABC):
    """
    Canonical AgentForge environment contract.

    This is intentionally aligned with the Gymnasium reset/step API while
    keeping the AgentForge contract explicit and framework-independent.
    """

    observation_space: gym.Space
    action_space: gym.Space

    def __init__(self) -> None:
        self._episode_done = False

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, Info]:
        """
        Start a new episode.

        Canonical signature:

            reset(
                *,
                seed: int | None = None,
                options: dict | None = None,
            ) -> tuple[Observation, Info]
        """
        raise NotImplementedError

    @abstractmethod
    def _step_impl(self, action: Action) -> StepResult:
        """Execute one environment transition."""
        raise NotImplementedError

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, Info]:
        """
        Canonical five-value environment step.

        Enforced invariants:

        - action must belong to action_space
        - reward must be finite
        - terminated/truncated cannot both be True
        - no step is allowed after episode end without reset
        - returned observation must belong to observation_space
        """
        if self._episode_done:
            raise RuntimeError(
                "step() called after episode end; call reset() first"
            )

        if not self.action_space.contains(action):
            raise ValueError(
                "action is not contained in action_space"
            )

        result = self._step_impl(action)

        if not self.observation_space.contains(result.observation):
            raise ValueError(
                "observation is not contained in observation_space"
            )

        if not isfinite(float(result.reward)):
            raise ValueError("reward must be finite")

        if result.terminated and result.truncated:
            raise ValueError(
                "terminated and truncated cannot both be True"
            )

        self._episode_done = (
            result.terminated or result.truncated
        )

        return (
            result.observation,
            float(result.reward),
            result.terminated,
            result.truncated,
            result.info,
        )

    def _reset_episode_state(self) -> None:
        """Mark the environment as active after reset."""
        self._episode_done = False
