from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentReport:
    task_id: str
    status: str
    planning_depth: int
    information_queries: int
    branch_decisions: int
    wrong_decisions: int
    delayed_failures: int
    recoveries: int
    replans: int
    redundant_actions: int
    invalid_actions: int
    total_cost: int
    final_reward: float
    terminal: bool
    success: bool

    @classmethod
    def from_state(
        cls,
        state,
        task_id: str | None = None,
        final_reward: float | None = None,
    ) -> "AgentReport":
        actions = list(getattr(state, "action_history", []))
        decisions = list(getattr(state, "decision_history", []))
        failures = list(getattr(state, "failure_history", []))
        recoveries = list(getattr(state, "recovery_history", []))

        information_queries = len(
            getattr(state, "observed_information", set())
        )

        branch_decisions = sum(
            1
            for action in actions
            if str(action).lower() in {
                "select_branch",
                "branch",
                "decisionbranch.select",
            }
        )

        wrong_decisions = sum(
            1
            for failure in failures
            if failure
        )

        delayed_failures = sum(
            1
            for failure in failures
            if isinstance(failure, dict)
            and failure.get("delayed", False)
        )

        redundant_actions = max(
            0,
            len(actions) - len(set(map(str, actions))),
        )

        invalid_actions = sum(
            1
            for action in actions
            if str(action).lower().startswith("invalid")
        )

        total_cost = int(
            round(
                100.0
                - float(getattr(state, "remaining_budget", 100.0))
            )
        )

        if final_reward is None:
            final_reward = 0.0

        total_reward = float(final_reward)

        planning_depth = max(
            0,
            int(getattr(state, "logical_stage", 0)),
        )

        return cls(
            task_id=str(
                task_id
                if task_id is not None
                else getattr(state, "task_id", "unknown")
            ),
            status=str(
                getattr(state, "system_status", "unknown")
            ),
            planning_depth=planning_depth,
            information_queries=information_queries,
            branch_decisions=branch_decisions,
            wrong_decisions=wrong_decisions,
            delayed_failures=delayed_failures,
            recoveries=len(recoveries),
            replans=sum(
                1
                for item in decisions
                if str(item).lower() == "replan"
            ),
            redundant_actions=redundant_actions,
            invalid_actions=invalid_actions,
            total_cost=total_cost,
            final_reward=total_reward,
            terminal=bool(getattr(state, "terminal", False)),
            success=bool(getattr(state, "success", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()
