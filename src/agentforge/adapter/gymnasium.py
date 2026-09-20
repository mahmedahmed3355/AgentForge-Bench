from __future__ import annotations

from typing import Any

import gymnasium as gym


class GymnasiumAdapter(gym.Env):
    """Adapt an AgentForge environment to the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: Any,
        *,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        self.environment = environment
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        result = self.environment.reset(seed=seed)

        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}

        if not isinstance(info, dict):
            info = dict(info)

        return observation, info

    def step(self, action: Any):
        result = self.environment.step(action)

        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError(
                "AgentForge environment.step() must return "
                "(observation, reward, terminated, truncated, info)."
            )

        observation, reward, terminated, truncated, info = result

        if not isinstance(info, dict):
            info = dict(info)

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        self.environment.close()
