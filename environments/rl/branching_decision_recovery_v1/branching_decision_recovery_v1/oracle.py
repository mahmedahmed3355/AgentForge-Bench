from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision_graph import DecisionBranch, DecisionStrategy


@dataclass(frozen=True)
class OracleResult:
    solved: bool
    valid_terminal_state: bool
    branch: str | None
    strategy: str | None
    logical_stage: int
    reason: str


class BranchingDecisionOracle:
    """Reference behavioral oracle.

    The oracle validates terminal behavior rather than requiring an exact
    action sequence. Historical failures are allowed when the final state
    demonstrates successful recovery/replanning and global validation.
    """

    MIN_LOGICAL_STAGE = 10

    def evaluate(self, state: Any) -> OracleResult:
        branch = state.selected_branch
        strategy = state.selected_strategy

        valid_branch = branch in {item.value for item in DecisionBranch}
        valid_strategy = strategy in {item.value for item in DecisionStrategy}

        no_active_failure = not state.failure_detected
        no_pending_recovery = not state.recovery_required
        no_pending_replan = not state.replanning_required
        globally_validated = state.validation_status == "globally_validated"

        terminal_conditions = (
            state.terminal,
            state.success,
            valid_branch,
            valid_strategy,
            no_active_failure,
            no_pending_recovery,
            no_pending_replan,
            globally_validated,
            state.logical_stage >= self.MIN_LOGICAL_STAGE,
        )

        solved = all(terminal_conditions)

        if solved:
            reason = "valid terminal trajectory"
        elif state.failure_detected:
            reason = "active failure remains unresolved"
        elif state.recovery_required or state.replanning_required:
            reason = "recovery or replanning remains pending"
        elif not globally_validated:
            reason = "global validation not completed"
        elif not state.terminal:
            reason = "episode has not reached terminal verification"
        else:
            reason = "terminal state does not satisfy oracle contract"

        return OracleResult(
            solved=solved,
            valid_terminal_state=solved,
            branch=branch,
            strategy=strategy,
            logical_stage=state.logical_stage,
            reason=reason,
        )

    def solve(self, env: Any, seed: int | None = None) -> OracleResult:
        """Execute a robust reference trajectory using the public API.

        This intentionally uses information gathering, hypothesis formation,
        decision selection, execution, recovery/replanning, and validation.
        It does not inspect hidden oracle fields.
        """
        if seed is None:
            env.reset()
        else:
            env.reset(seed=seed)

        env.step("inspect_system")
        env.step("inspect_component", component="processing")
        env.step("inspect_dependency", dependency="processing")
        env.step("inspect_history")

        env.step(
            "form_hypothesis",
            hypothesis="downstream_dependency_risk",
            confidence=0.75,
        )
        env.step("select_branch", branch="branch_a")
        env.step("select_strategy", strategy="strategy_a")
        env.step("apply_action")

        # Execute until the environment actually exposes a delayed
        # consequence. The oracle must not assume a fixed delay because
        # hidden scenarios may vary consequence timing.
        max_execution_steps = 32

        for _ in range(max_execution_steps):
            if env.state.terminal or env.state.failure_detected:
                break
            env.step("run")

        if env.state.failure_detected:
            env.step("recover", strategy="rollback", cost=8.0)
            env.step("replan")

            # The reference trajectory intentionally selects a stable
            # alternative after replanning rather than reusing the failed
            # branch.
            env.step("select_branch", branch="branch_b")
            env.step("select_strategy", strategy="strategy_b")
            env.step("apply_action")

            # Re-execute after replanning. Branch B has no delayed
            # consequence in the reference graph, but real execution is
            # still required before validation.
            for _ in range(4):
                if env.state.terminal:
                    break
                env.step("run")

            if not env.state.terminal:
                env.step("validate")

            # A local validation result is not equivalent to global
            # validation. Continue through the public execution interface
            # until the environment itself reaches global validation.
            if (
                not env.state.terminal
                and env.state.validation_status != "globally_validated"
            ):
                env.step("validate")

        env.step("final_verify")

        return self.evaluate(env.state)
