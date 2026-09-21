"""Gymnasium adapter for native AgentForge environments.

Architecture:

    Native AgentForge Environment
                |
                v
         GymnasiumAdapter
                |
                v
          gymnasium.Env

The native environment remains responsible for AgentForge framework
semantics. This adapter exposes the standard Gymnasium interface without
forcing the native implementation to inherit from Gymnasium.
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
from gymnasium import spaces


class NativeEnvironment(Protocol):
    """Protocol required by the Gymnasium adapter."""

    action_space: spaces.Space[Any]
    observation_space: spaces.Space[Any]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        ...

    def render(self) -> Any:
        ...

    def close(self) -> None:
        ...


class GymnasiumAdapter(gym.Env[Any, Any]):
    """Expose a native AgentForge environment as gymnasium.Env."""

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        environment: NativeEnvironment,
    ) -> None:
        if environment is None:
            raise ValueError(
                "environment must not be None."
            )

        if not hasattr(environment, "action_space"):
            raise TypeError(
                "Native environment must define action_space."
            )

        if not hasattr(environment, "observation_space"):
            raise TypeError(
                "Native environment must define observation_space."
            )

        self._environment = environment

        self.action_space = environment.action_space
        self.observation_space = environment.observation_space

        native_metadata = getattr(
            environment,
            "metadata",
            None,
        )

        if isinstance(native_metadata, dict):
            self.metadata = dict(native_metadata)

    @property
    def native_environment(self) -> NativeEnvironment:
        """Return the wrapped native AgentForge environment."""
        return self._environment

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the native environment using Gymnasium's contract."""

        super().reset(seed=seed)

        observation, info = self._environment.reset(
            seed=seed,
            options=options,
        )

        if not self.observation_space.contains(observation):
            raise ValueError(
                "Native environment returned an observation outside "
                "observation_space."
            )

        if not isinstance(info, dict):
            raise TypeError(
                "Native environment reset() must return "
                "(observation, dict_info)."
            )

        return observation, info

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one native environment step."""

        if not self.action_space.contains(action):
            raise ValueError(
                "Action is not contained in action_space."
            )

        observation, reward, terminated, truncated, info = (
            self._environment.step(action)
        )

        if not self.observation_space.contains(observation):
            raise ValueError(
                "Native environment returned an observation outside "
                "observation_space."
            )

        if not isinstance(reward, (int, float)) or isinstance(
            reward,
            bool,
        ):
            raise TypeError(
                "Native environment reward must be a numeric scalar."
            )

        if not isinstance(terminated, bool):
            raise TypeError(
                "terminated must be bool."
            )

        if not isinstance(truncated, bool):
            raise TypeError(
                "truncated must be bool."
            )

        if terminated and truncated:
            raise ValueError(
                "terminated and truncated cannot both be True."
            )

        if not isinstance(info, dict):
            raise TypeError(
                "Native environment step() must return "
                "(observation, reward, terminated, truncated, dict_info)."
            )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> Any:
        """Delegate rendering to the native environment."""
        return self._environment.render()

    def close(self) -> None:
        """Close the native environment."""
        self._environment.close()
