from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

TEXT_CHARSET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "_-:. "
)

from .actions import ActionKind
from .environment import BranchingDecisionRecoveryEnv
from .observations import Observation, validate_observation_contract


class BranchingDecisionRecoveryGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario: dict[str, Any] | None = None):
        super().__init__()
        self.scenario = scenario or {
            "scenario_id": "decision-v1-gym-default",
            "seed": 3000,
            "scenario_family": "dependency",
        }

        self.native_env = BranchingDecisionRecoveryEnv()

        self.terminated = False
        self.truncated = False

        self.observation_space = gym.spaces.Dict(
            {
                "step": gym.spaces.Box(
                    low=0.0,
                    high=1000.0,
                    shape=(),
                    dtype=np.float32,
                ),
                "logical_stage": gym.spaces.Box(
                    low=0.0,
                    high=1000.0,
                    shape=(),
                    dtype=np.float32,
                ),
                "system_status": gym.spaces.Text(
                    min_length=0,
                    max_length=64,
                    charset=TEXT_CHARSET,
                ),
                "current_component": gym.spaces.Text(
                    min_length=0,
                    max_length=64,
                    charset=TEXT_CHARSET,
                ),
                "visible_components": gym.spaces.Sequence(
                    gym.spaces.Text(
                    min_length=0,
                    max_length=64,
                    charset=TEXT_CHARSET,
                ),
                ),
                "available_actions": gym.spaces.Sequence(
                    gym.spaces.Text(
                    min_length=0,
                    max_length=64,
                    charset=TEXT_CHARSET,
                ),
                ),
                "observed_information": gym.spaces.Sequence(
                    gym.spaces.Text(
                    min_length=0,
                    max_length=128,
                    charset=TEXT_CHARSET,
                ),
                ),
                "warnings": gym.spaces.Sequence(
                    gym.spaces.Text(
                    min_length=0,
                    max_length=128,
                    charset=TEXT_CHARSET,
                ),
                ),
                "local_metrics": gym.spaces.Dict(
                    {
                        "latency_ms": gym.spaces.Box(
                            low=0.0,
                            high=1_000_000.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "error_rate": gym.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "resource_usage": gym.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "hypothesis": gym.spaces.Text(
                    min_length=0,
                    max_length=128,
                    charset=TEXT_CHARSET,
                ),
                "hypothesis_confidence": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(),
                    dtype=np.float32,
                ),
                "selected_branch": gym.spaces.Text(
                    min_length=0,
                    max_length=64,
                    charset=TEXT_CHARSET,
                ),
                "selected_strategy": gym.spaces.Text(
                    min_length=0,
                    max_length=64,
                    charset=TEXT_CHARSET,
                ),
                "failure_detected": gym.spaces.Discrete(2),
                "recovery_required": gym.spaces.Discrete(2),
                "replanning_required": gym.spaces.Discrete(2),
                "terminal": gym.spaces.Discrete(2),
                "success": gym.spaces.Discrete(2),
            }
        )

        self.action_space = gym.spaces.Discrete(len(ActionKind))

    def _observation(self) -> Observation:
        state = self.native_env.state

        obs = Observation(
            step=int(getattr(state, "step", 0)),
            logical_stage=int(getattr(state, "logical_stage", 0)),
            system_status=str(getattr(state, "system_status", "unknown")),
            current_component=getattr(state, "current_component", None),
            visible_components=tuple(
                getattr(state, "component_states", {}).keys()
            ),
            available_actions=tuple(
                action.value for action in ActionKind
            ),
            observed_information=tuple(
                getattr(state, "observed_information", [])
            ),
            warnings=tuple(
                getattr(state, "warnings", [])
            ),
            local_metrics=dict(
                getattr(state, "local_metrics", {})
            ),
            hypothesis=getattr(state, "hypothesis", None),
            hypothesis_confidence=float(
                getattr(state, "hypothesis_confidence", 0.0)
            ),
            selected_branch=getattr(state, "selected_branch", None),
            selected_strategy=getattr(state, "selected_strategy", None),
            failure_detected=bool(
                getattr(state, "failure_detected", False)
            ),
            recovery_required=bool(
                getattr(state, "recovery_required", False)
            ),
            replanning_required=bool(
                getattr(state, "replanning_required", False)
            ),
            terminal=bool(getattr(state, "terminal", False)),
            success=bool(getattr(state, "success", False)),
        )

        validate_observation_contract(obs)
        return obs

    def _validate_gym_observation(self, observation: dict[str, Any]) -> None:
        if not self.observation_space.contains(observation):
            for key, space in self.observation_space.spaces.items():
                value = observation.get(key)
                if not space.contains(value):
                    raise AssertionError(
                        f"Gym observation field {key!r} is outside its space. "
                        f"value={value!r}, space={space!r}"
                    )
            raise AssertionError(
                "Gym observation is outside observation_space."
            )

    def _to_gym(self, obs: Observation) -> dict[str, Any]:
        return {
            "step": np.float32(obs.step),
            "logical_stage": np.float32(obs.logical_stage),
            "system_status": obs.system_status,
            "current_component": obs.current_component or "",
            "visible_components": tuple(obs.visible_components),
            "available_actions": tuple(obs.available_actions),
            "observed_information": tuple(obs.observed_information),
            "warnings": tuple(obs.warnings),
            "local_metrics": {
                "latency_ms": np.float32(
                    obs.local_metrics.get("latency_ms", 0.0)
                ),
                "error_rate": np.float32(
                    obs.local_metrics.get("error_rate", 0.0)
                ),
                "resource_usage": np.float32(
                    obs.local_metrics.get("resource_usage", 0.0)
                ),
            },
            "hypothesis": obs.hypothesis or "",
            "hypothesis_confidence": np.float32(
                obs.hypothesis_confidence
            ),
            "selected_branch": obs.selected_branch or "",
            "selected_strategy": obs.selected_strategy or "",
            "failure_detected": int(obs.failure_detected),
            "recovery_required": int(obs.recovery_required),
            "replanning_required": int(obs.replanning_required),
            "terminal": int(obs.terminal),
            "success": int(obs.success),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.scenario = {
                **self.scenario,
                "seed": int(seed),
            }

        self.native_env.reset()
        obs = self._observation()

        self.terminated = False
        self.truncated = False

        observation = self._to_gym(obs)
        self._validate_gym_observation(observation)

        return observation, {
            "scenario_id": self.scenario["scenario_id"],
            "seed": self.scenario["seed"],
        }

    def step(self, action: int):
        action_kind = list(ActionKind)[int(action)]

        result = self.native_env.step(action_kind)

        if not isinstance(result, tuple) or len(result) != 4:
            raise TypeError(
                f"Native environment step must return "
                f"(observation, reward, terminal, info), got {type(result).__name__}"
            )

        native_observation, native_reward, native_terminal, native_info = result

        obs = self._observation()
        info = dict(native_info)
        info.update(
            {
                "action": action_kind.value,
                "logical_stage": obs.logical_stage,
                "terminal": obs.terminal,
                "success": obs.success,
            }
        )

        self.terminated = bool(obs.terminal and obs.success)
        self.truncated = bool(obs.terminal and not obs.success)

        observation = self._to_gym(obs)
        self._validate_gym_observation(observation)

        return (
            observation,
            float(native_reward),
            self.terminated,
            self.truncated,
            info,
        )

    def render(self):
        return self._observation().to_dict()

    def close(self):
        return None
