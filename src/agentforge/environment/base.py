from __future__ import annotations

from abc import abstractmethod
from typing import Any

import gymnasium as gym

from agentforge.contracts.environment import (
    EnvironmentContract,
    Info,
    Observation,
)
from agentforge.environment.state import EnvironmentState
from agentforge.environment.transition import Transition


class AgentForgeEnv(gym.Env, EnvironmentContract):
    """
    Canonical AgentForge reinforcement-learning environment.

    Architecture:

        AgentForgeEnv
              |
              +--> gymnasium.Env
              |
              +--> EnvironmentContract

    The public lifecycle is owned by this class. Concrete tasks implement
    only the two task-specific hooks:

        _reset_impl(seed=..., options=...)
        _step_impl(action)

    This keeps task logic separate from the framework lifecycle and ensures
    every environment follows the same Gymnasium-compatible contract.
    """

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()

        self._state: EnvironmentState | None = None
        self._episode_counter = 0

    @property
    def state(self) -> EnvironmentState | None:
        return self._state

    @property
    def episode_id(self) -> str | None:
        if self._state is None:
            return None
        return self._state.episode_id

    @property
    def terminated(self) -> bool:
        return bool(self._state and self._state.terminated)

    @property
    def truncated(self) -> bool:
        return bool(self._state and self._state.truncated)

    @property
    def done(self) -> bool:
        return bool(self._state and self._state.done)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, Info]:
        """
        Start a new episode.

        This method owns the framework lifecycle. Concrete environments
        provide task-specific reset behavior through _reset_impl().
        """
        super().reset(seed=seed)

        self._episode_counter += 1

        episode_id = self._create_episode_id(seed)

        self._state = EnvironmentState(
            episode_id=episode_id,
            step_count=0,
            terminated=False,
            truncated=False,
            payload={},
        )

        observation, info = self._reset_impl(
            seed=seed,
            options=options,
        )

        self._validate_observation(observation)

        if not isinstance(info, dict):
            raise TypeError(
                "_reset_impl() must return "
                "(observation, info) where info is a dict"
            )

        return observation, info

    def step(
        self,
        action: Any,
    ) -> tuple[Observation, float, bool, bool, Info]:
        """
        Execute one environment transition.

        Concrete environments provide task-specific behavior through
        _step_impl(). The framework owns lifecycle validation.
        """
        self._ensure_episode_active()
        self._validate_action(action)

        transition = self._step_impl(action)

        if not isinstance(transition, Transition):
            raise TypeError(
                "_step_impl() must return Transition"
            )

        self._validate_transition(transition)

        assert self._state is not None

        self._state.step_count += 1

        if transition.terminated:
            self._state.mark_terminated()
        elif transition.truncated:
            self._state.mark_truncated()

        return (
            transition.observation,
            float(transition.reward),
            bool(transition.terminated),
            bool(transition.truncated),
            transition.info,
        )

    def close(self) -> None:
        self._state = None
        super().close()

    def _ensure_episode_active(self) -> None:
        if self._state is None:
            raise RuntimeError(
                "environment must be reset before step()"
            )

        if self._state.done:
            raise RuntimeError(
                "cannot call step() after terminated/truncated; "
                "call reset() first"
            )

    def _mark_episode_end(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if terminated and truncated:
            raise ValueError(
                "terminated and truncated cannot both be True"
            )

        if self._state is None:
            raise RuntimeError(
                "environment must be reset before marking episode end"
            )

        if terminated:
            self._state.mark_terminated()
        elif truncated:
            self._state.mark_truncated()

    def _validate_action(self, action: Any) -> None:
        if not self.action_space.contains(action):
            raise ValueError(
                "action is not contained in action_space"
            )

    def _validate_observation(self, observation: Any) -> None:
        if not self.observation_space.contains(observation):
            raise ValueError(
                "observation is not contained in observation_space"
            )

    def _validate_transition(
        self,
        transition: Transition,
    ) -> None:
        if transition.terminated and transition.truncated:
            raise ValueError(
                "terminated and truncated cannot both be True"
            )

        reward = transition.reward

        if not isinstance(reward, (int, float)):
            raise TypeError(
                "reward must be a finite scalar"
            )

        if not isinstance(reward, bool):
            import math

            if not math.isfinite(float(reward)):
                raise ValueError(
                    "reward must be a finite scalar"
                )

        self._validate_observation(transition.observation)

        if not isinstance(transition.info, dict):
            raise TypeError(
                "transition info must be a dict"
            )

    def _create_episode_id(self, seed: int | None) -> str:
        if seed is None:
            return (
                f"episode-{id(self)}-{self._episode_counter}"
            )

        return (
            f"episode-{seed}-{self._episode_counter}"
        )

    @abstractmethod
    def _reset_impl(
        self,
        *,
        seed: int | None,
        options: dict[str, Any] | None,
    ) -> tuple[Observation, Info]:
        """
        Task-specific reset hook.

        Implementations must return:
            (observation, info)
        """
        raise NotImplementedError

    @abstractmethod
    def _step_impl(
        self,
        action: Any,
    ) -> Transition:
        """
        Task-specific transition hook.

        Implementations must return a Transition containing:
            observation
            reward
            terminated
            truncated
            info
        """
        raise NotImplementedError
