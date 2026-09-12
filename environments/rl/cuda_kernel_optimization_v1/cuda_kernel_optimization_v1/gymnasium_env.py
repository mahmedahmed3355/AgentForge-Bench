from __future__ import annotations

from typing import Any

from .environment import CudaKernelOptimizationEnvironment
from .report import EpisodeReporter


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    gym = None
    spaces = None
    _GYMNASIUM_IMPORT_ERROR = exc
else:
    _GYMNASIUM_IMPORT_ERROR = None


ACTION_NAMES = (
    "inspect",
    "modify_memory",
    "modify_tiling",
    "modify_compute",
    "compile",
    "check_correctness",
    "benchmark",
    "analyze",
    "recover",
    "final_verify",
)


def _require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise ImportError(
            "Gymnasium is not installed. "
            "Install the environment with Gymnasium support "
            "before using CudaKernelOptimizationGymEnv."
        ) from _GYMNASIUM_IMPORT_ERROR


def _serialize_observation(state: Any) -> dict[str, Any]:
    if hasattr(state, "to_dict") and callable(state.to_dict):
        data = state.to_dict()
    elif hasattr(state, "__dataclass_fields__"):
        data = {
            name: getattr(state, name)
            for name in state.__dataclass_fields__
        }
    elif isinstance(state, dict):
        data = dict(state)
    else:
        raise TypeError(
            f"Unsupported environment observation type: {type(state)!r}"
        )

    result: dict[str, Any] = {}

    for key, value in data.items():
        if hasattr(value, "value"):
            value = value.value

        if isinstance(value, list):
            value = [
                item.value if hasattr(item, "value") else item
                for item in value
            ]

        result[key] = value

    return result


if gym is not None:

    class CudaKernelOptimizationGymEnv(gym.Env):
        """
        Gymnasium adapter for AgentForge Task 001.

        The underlying AgentForge environment remains the source of truth.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            seed: int | None = None,
            task_id: str = "cuda-kernel-optimization-v1",
            scenario_id: str | None = None,
        ):
            super().__init__()

            self.task_id = task_id
            self.scenario_id = scenario_id
            self.seed_value = seed

            self._env = CudaKernelOptimizationEnvironment(seed=seed)

            self.action_space = spaces.Discrete(len(ACTION_NAMES))

            self.observation_space = spaces.Dict(
                {
                    "logical_stage": spaces.Discrete(21),
                    "action_count": spaces.Discrete(33),
                    "inspected": spaces.Discrete(2),
                    "candidate_modified": spaces.Discrete(2),
                    "compilation_ok": spaces.Discrete(2),
                    "correctness_ok": spaces.Discrete(2),
                    "benchmark_completed": spaces.Discrete(2),
                    "analysis_completed": spaces.Discrete(2),
                    "performance_improved": spaces.Discrete(2),
                    "regression_detected": spaces.Discrete(2),
                    "recovery_count": spaces.Discrete(33),
                    "terminal": spaces.Discrete(2),
                    "success": spaces.Discrete(2),
                }
            )

            self.reporter = EpisodeReporter(
                task_id=task_id,
                scenario_id=scenario_id,
                seed=seed,
            )

        def _observation(self) -> dict[str, Any]:
            state = self._env.state
            data = _serialize_observation(state)

            allowed = set(self.observation_space.spaces)

            return {
                key: int(bool(value))
                if isinstance(value, bool)
                else int(value)
                for key, value in data.items()
                if key in allowed
            }

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ):
            super().reset(seed=seed)

            if seed is not None:
                self.seed_value = seed

            state = self._env.reset(seed=seed)

            self.reporter = EpisodeReporter(
                task_id=self.task_id,
                scenario_id=self.scenario_id,
                seed=self.seed_value,
            )
            self.reporter.start()

            observation = self._observation()

            info = {
                "task_id": self.task_id,
                "scenario_id": self.scenario_id,
                "seed": self.seed_value,
                "reporter": self.reporter,
            }

            return observation, info

        def step(self, action: int):
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid Gymnasium action: {action}")

            action_name = ACTION_NAMES[int(action)]

            observation, reward, terminal, info = self._env.step(
                action_name
            )

            gym_observation = self._observation()

            self.reporter.record_step(
                action=action_name,
                reward=float(reward),
                observation=gym_observation,
                terminal=bool(terminal),
                success=bool(gym_observation.get("success", False)),
                info=info,
            )

            terminated = bool(terminal)
            truncated = bool(
                gym_observation.get("action_count", 0) >= 32
                and not terminated
            )

            info = dict(info)
            info["task_id"] = self.task_id
            info["action_name"] = action_name

            if terminated or truncated:
                self.reporter.finalize(
                    success=bool(gym_observation.get("success", False)),
                    status=(
                        "success"
                        if gym_observation.get("success", False)
                        else "failure"
                    ),
                )

            return (
                gym_observation,
                float(reward),
                terminated,
                truncated,
                info,
            )

        def render(self):
            return self._observation()

        def close(self):
            return None

        @property
        def agentforge_env(self):
            return self._env

        @property
        def episode_report(self):
            return self.reporter.report

else:

    class CudaKernelOptimizationGymEnv:
        def __init__(self, *args, **kwargs):
            _require_gymnasium()
