from __future__ import annotations

from typing import Any

from verifiers.v1 import tool
from verifiers.v1.mcp import Toolset, ToolsetConfig

from .actions import ActionKind
from .environment import BranchingDecisionRecoveryEnv
from .prime_state import BranchingDecisionPrimeState


def _sync_state(
    state: BranchingDecisionPrimeState,
    native_state,
    observation: Any,
) -> BranchingDecisionPrimeState:
    state.seed = native_state.seed
    state.scenario_id = native_state.episode_id
    state.logical_stage = native_state.logical_stage
    state.step = native_state.step
    state.terminal = native_state.terminal
    state.success = native_state.success
    state.last_observation = str(observation)
    return state


class BranchingDecisionToolset(
    Toolset[BranchingDecisionPrimeState]
):
    """Prime V1 toolset for Branching Decision & Recovery V1."""

    TOOL_PREFIX = "branching"

    def __init__(self, config: ToolsetConfig):
        super().__init__(config)

        self._env = BranchingDecisionRecoveryEnv()

        self._task_seed: int | None = None
        self._task_scenario_id: str | None = None

    async def setup_task(self, task) -> None:
        self._task_seed = getattr(
            task.data,
            "seed",
            None,
        )

        self._task_scenario_id = getattr(
            task.data,
            "scenario_id",
            None,
        )

    def _run_action(
        self,
        action: ActionKind,
        **kwargs: Any,
    ) -> dict:
        state = self.state

        if state is None:
            raise RuntimeError(
                "Prime toolset state is not initialized."
            )

        if state.step == 0:
            seed = (
                state.seed
                if state.seed is not None
                else self._task_seed
            )

            scenario_id = (
                self._task_scenario_id
                if self._task_scenario_id is not None
                else "decision-v1-public-001"
            )

            observation = self._env.reset(
                scenario_id=scenario_id,
                seed=seed,
            )

            state.seed = seed
        else:
            observation = None

        observation, reward, done, info = self._env.step(
            action,
            **kwargs,
        )

        native_state = self._env.state

        if native_state is None:
            raise RuntimeError(
                "Native environment state is unavailable."
            )

        _sync_state(
            state=state,
            native_state=native_state,
            observation=observation,
        )

        safe_info = {
            str(key): (
                value.value
                if hasattr(value, "value")
                else value
            )
            for key, value in info.items()
        }

        return {
            "action": action.value,
            "reward": float(reward),
            "done": bool(done),
            "success": bool(native_state.success),
            "logical_stage": native_state.logical_stage,
            "step": native_state.step,
            "observation": str(observation),
            "info": safe_info,
        }

    @tool
    def inspect_system(self) -> dict:
        return self._run_action(
            ActionKind.INSPECT_SYSTEM
        )

    @tool
    def inspect_component(
        self,
        component: str = "processing",
    ) -> dict:
        return self._run_action(
            ActionKind.INSPECT_COMPONENT,
            component=component,
        )

    @tool
    def inspect_dependency(
        self,
        dependency: str = "primary_dependency",
    ) -> dict:
        return self._run_action(
            ActionKind.INSPECT_DEPENDENCY,
            dependency=dependency,
        )

    @tool
    def inspect_history(self) -> dict:
        return self._run_action(
            ActionKind.INSPECT_HISTORY
        )

    @tool
    def probe_state(self) -> dict:
        return self._run_action(
            ActionKind.PROBE_STATE
        )

    @tool
    def query_validation(self) -> dict:
        return self._run_action(
            ActionKind.QUERY_VALIDATION
        )

    @tool
    def form_hypothesis(
        self,
        hypothesis: str,
        confidence: float = 0.0,
    ) -> dict:
        return self._run_action(
            ActionKind.FORM_HYPOTHESIS,
            hypothesis=hypothesis,
            confidence=confidence,
        )

    @tool
    def select_branch(
        self,
        branch: str,
    ) -> dict:
        return self._run_action(
            ActionKind.SELECT_BRANCH,
            branch=branch,
        )

    @tool
    def select_strategy(
        self,
        strategy: str,
    ) -> dict:
        return self._run_action(
            ActionKind.SELECT_STRATEGY,
            strategy=strategy,
        )

    @tool
    def apply_action(
        self,
        cost: float | None = None,
    ) -> dict:
        kwargs = {}

        if cost is not None:
            kwargs["cost"] = cost

        return self._run_action(
            ActionKind.APPLY_ACTION,
            **kwargs,
        )

    @tool
    def modify_configuration(
        self,
        cost: float = 4.0,
    ) -> dict:
        return self._run_action(
            ActionKind.MODIFY_CONFIGURATION,
            cost=cost,
        )

    @tool
    def run(self) -> dict:
        return self._run_action(
            ActionKind.RUN
        )

    @tool
    def validate(self) -> dict:
        return self._run_action(
            ActionKind.VALIDATE
        )

    @tool
    def recover(
        self,
        strategy: str = "reconfigure",
        cost: float = 8.0,
    ) -> dict:
        return self._run_action(
            ActionKind.RECOVER,
            strategy=strategy,
            cost=cost,
        )

    @tool
    def replan(self) -> dict:
        return self._run_action(
            ActionKind.REPLAN
        )

    @tool
    def final_verify(self) -> dict:
        return self._run_action(
            ActionKind.FINAL_VERIFY
        )
