from __future__ import annotations

from typing import Any

from .actions import ACTION_NAMES
from .environment import DataPipelineRecoveryEnv
from .decision_graph import IncidentKind


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    gym = None
    spaces = None
    _GYMNASIUM_IMPORT_ERROR = exc
else:
    _GYMNASIUM_IMPORT_ERROR = None


def _require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise ImportError(
            "Gymnasium is not installed. Install this environment with "
            "Gymnasium support before using DataPipelineRecoveryGymEnv."
        ) from _GYMNASIUM_IMPORT_ERROR


def _observation_dict(observation) -> dict[str, Any]:
    data = {}

    for name in observation.__dataclass_fields__:
        value = getattr(observation, name)

        if hasattr(value, "value"):
            value = value.value

        if isinstance(value, set):
            value = sorted(value)

        data[name] = value

    return data


if gym is not None:

    class DataPipelineRecoveryGymEnv(gym.Env):
        """Gymnasium adapter for AgentForge Task 002."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            seed: int | None = None,
            incident: str = "schema",
            task_id: str = "data-pipeline-recovery-v1",
            scenario_id: str | None = None,
        ):
            super().__init__()

            self.task_id = task_id
            self.scenario_id = scenario_id
            self.seed_value = seed
            self.incident = IncidentKind(incident)

            self._env = DataPipelineRecoveryEnv(
                episode_id="gymnasium-episode",
                incident=self.incident,
            )

            self.action_space = spaces.Discrete(len(ACTION_NAMES))

            self.observation_space = spaces.Dict(
                {
                    "logical_stage": spaces.Discrete(41),
                    "action_count": spaces.Discrete(41),
                    "incident_observed": spaces.Discrete(2),
                    "diagnosis_started": spaces.Discrete(2),
                    "root_cause_identified": spaces.Discrete(2),
                    "schema_valid": spaces.Discrete(2),
                    "cleaning_valid": spaces.Discrete(2),
                    "transformation_valid": spaces.Discrete(2),
                    "aggregation_valid": spaces.Discrete(2),
                    "output_valid": spaces.Discrete(2),
                    "pipeline_run_completed": spaces.Discrete(2),
                    "failure_detected": spaces.Discrete(2),
                    "recovery_required": spaces.Discrete(2),
                    "recovery_completed": spaces.Discrete(2),
                    "replanning_required": spaces.Discrete(2),
                    "downstream_inconsistency": spaces.Discrete(2),
                    "global_validation_completed": spaces.Discrete(2),
                    "terminal": spaces.Discrete(2),
                    "success": spaces.Discrete(2),
                }
            )

        def _observation(self):
            raw = _observation_dict(self._env.state)

            return {
                key: int(bool(value))
                if isinstance(value, bool)
                else int(value)
                for key, value in raw.items()
                if key in self.observation_space.spaces
            }

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)

            if seed is not None:
                self.seed_value = seed

            self._env.reset()

            return self._observation(), {
                "task_id": self.task_id,
                "scenario_id": self.scenario_id,
                "seed": self.seed_value,
            }

        def step(self, action: int):
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid Gymnasium action: {action}")

            action_name = ACTION_NAMES[int(action)]

            result = self._env.step(
                __import__(
                    "data_pipeline_recovery_v1.actions",
                    fromlist=["ActionKind"],
                ).ActionKind(action_name)
            )

            observation = self._observation()

            terminated = bool(result.terminated)
            truncated = bool(result.truncated)

            info = dict(result.info)
            info["task_id"] = self.task_id
            info["scenario_id"] = self.scenario_id
            info["action_name"] = action_name

            return (
                observation,
                float(result.reward),
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

else:

    class DataPipelineRecoveryGymEnv:
        def __init__(self, *args, **kwargs):
            _require_gymnasium()
