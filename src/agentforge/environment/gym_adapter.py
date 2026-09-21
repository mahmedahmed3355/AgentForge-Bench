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


        self._environment = environment

        # Preserve compatibility with native environments that do not expose
        # explicit Gymnasium spaces. Expose spaces only when provided.
        self.action_space = getattr(environment, "action_space", None)
        self.observation_space = getattr(environment, "observation_space", None)

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
        "Reset the wrapped native environment using the Gymnasium contract."

        super().reset(seed=seed)

        reset_method = self._environment.reset

        if options is not None:
            try:
                result = reset_method(seed=seed, options=options)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                result = reset_method(seed=seed)
        else:
            try:
                result = reset_method(seed=seed)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                result = reset_method()

        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}

        if self.observation_space is not None and not self.observation_space.contains(observation):
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
        "Execute one native environment step using Gymnasium semantics."

        if self.action_space is not None and not self.action_space.contains(action):
            raise ValueError(
                "Action is not contained in action_space."
            )

        result = self._environment.step(action)

        if not isinstance(result, tuple):
            raise TypeError(
                "Native environment step() must return a tuple."
            )

        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raise ValueError(
                "Native environment step() must return 4 or 5 values."
            )

        if self.observation_space is not None and not self.observation_space.contains(observation):
            raise ValueError(
                "Native environment returned an observation outside "
                "observation_space."
            )

        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise TypeError(
                "Native environment reward must be a numeric scalar."
            )

        if not isinstance(terminated, bool):
            raise TypeError("terminated must be bool.")

        if not isinstance(truncated, bool):
            raise TypeError("truncated must be bool.")

        if terminated and truncated:
            raise ValueError(
                "terminated and truncated cannot both be True."
            )

        if not isinstance(info, dict):
            raise TypeError(
                "Native environment step() must return dict info."
            )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Close the native environment."""
        self._environment.close()
