from __future__ import annotations

from verifiers.v1 import tool
from verifiers.v1.mcp import Toolset, ToolsetConfig

from .actions import ActionKind
from .environment import DataPipelineRecoveryEnv
from .prime_state import PipelinePrimeState


class PipelineRecoveryToolset(Toolset[PipelinePrimeState]):
    """Prime V1 tools for Data Pipeline Recovery V1."""

    TOOL_PREFIX = "pipeline"

    def __init__(self, config: ToolsetConfig):
        super().__init__(config)
        self._env = DataPipelineRecoveryEnv()
        self._task_seed: int | None = None

    async def setup_task(self, task) -> None:
        self._task_seed = getattr(task.data, "seed", None)

    def _sync_state(self, result) -> None:
        native = result.observation
        state = self.state

        state.incident_observed = native.incident_observed
        state.diagnosis_started = native.diagnosis_started
        state.root_cause_identified = native.root_cause_identified

        state.selected_branch = native.selected_branch
        state.selected_strategy = native.selected_strategy

        state.schema_valid = native.schema_valid
        state.cleaning_valid = native.cleaning_valid
        state.transformation_valid = native.transformation_valid
        state.aggregation_valid = native.aggregation_valid
        state.output_valid = native.output_valid

        state.pipeline_run_completed = native.pipeline_run_completed
        state.failure_detected = native.failure_detected

        state.recovery_required = native.recovery_required
        state.recovery_completed = native.recovery_completed

        state.replanning_required = native.replanning_required

        state.downstream_inconsistency = native.downstream_inconsistency
        state.global_validation_completed = native.global_validation_completed

        state.logical_stage = native.logical_stage
        state.action_count = native.action_count
        state.terminal = native.terminal
        state.success = native.success

        observation = repr(native)
        state.last_observation = observation
        state.history.append(observation)

    def _run_action(self, action: ActionKind, **kwargs) -> dict:
        if self.state.action_count == 0:
            self._env.reset()

        result = self._env.step(action, **kwargs)
        self._sync_state(result)

        return {
            "action": action.value,
            "reward": float(result.reward),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "success": bool(result.observation.success),
            "logical_stage": result.observation.logical_stage,
            "action_count": result.observation.action_count,
            "observation": repr(result.observation),
            "info": dict(result.info),
        }

    @tool
    def inspect_pipeline(self) -> dict:
        return self._run_action(ActionKind.INSPECT_PIPELINE)

    @tool
    def inspect_component(self, component: str) -> dict:
        return self._run_action(
            ActionKind.INSPECT_COMPONENT,
            component=component,
        )

    @tool
    def analyze_failure(self) -> dict:
        return self._run_action(ActionKind.ANALYZE_FAILURE)

    @tool
    def select_branch(self, branch: str) -> dict:
        return self._run_action(
            ActionKind.SELECT_BRANCH,
            branch=branch,
        )

    @tool
    def select_strategy(self, strategy: str) -> dict:
        return self._run_action(
            ActionKind.SELECT_STRATEGY,
            strategy=strategy,
        )

    @tool
    def modify_component(self, component: str = "") -> dict:
        return self._run_action(
            ActionKind.MODIFY_COMPONENT,
            component=component,
        )

    @tool
    def modify_config(self) -> dict:
        return self._run_action(ActionKind.MODIFY_CONFIG)

    @tool
    def run_pipeline(self) -> dict:
        return self._run_action(ActionKind.RUN_PIPELINE)

    @tool
    def check_schema(self) -> dict:
        return self._run_action(ActionKind.CHECK_SCHEMA)

    @tool
    def check_output(self) -> dict:
        return self._run_action(ActionKind.CHECK_OUTPUT)

    @tool
    def recover(self) -> dict:
        return self._run_action(ActionKind.RECOVER)

    @tool
    def replan(self) -> dict:
        return self._run_action(ActionKind.REPLAN)

    @tool
    def final_verify(self) -> dict:
        return self._run_action(ActionKind.FINAL_VERIFY)
