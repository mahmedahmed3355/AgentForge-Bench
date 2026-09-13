from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentReport:
    task_id: str
    incident: str
    status: str
    actions: int
    logical_stages: int
    total_reward: float
    solved: tuple[str, ...]
    incomplete: tuple[str, ...]
    recovery_required: bool
    recovery_completed: bool
    replanning_required: bool
    downstream_inconsistency: bool
    stopping_point: str
    failure_reason: str
    terminal: bool
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "incident": self.incident,
            "status": self.status,
            "actions": self.actions,
            "logical_stages": self.logical_stages,
            "total_reward": self.total_reward,
            "solved": self.solved,
            "incomplete": self.incomplete,
            "recovery_required": self.recovery_required,
            "recovery_completed": self.recovery_completed,
            "replanning_required": self.replanning_required,
            "downstream_inconsistency": self.downstream_inconsistency,
            "stopping_point": self.stopping_point,
            "failure_reason": self.failure_reason,
            "terminal": self.terminal,
            "success": self.success,
        }

    def render(self) -> str:
        lines = [
            "AgentForge-Bench — Task 002 Agent Report",
            "",
            f"Task: {self.task_id}",
            f"Incident: {self.incident}",
            "",
            f"Status: {self.status}",
            "",
            "Steps:",
            f"  Actions: {self.actions}",
            f"  Logical stages: {self.logical_stages}",
            f"  Total reward: {self.total_reward:.2f}",
            "",
            "Progress:",
        ]

        for item in self.solved:
            lines.append(f"  ✓ {item}")

        for item in self.incomplete:
            lines.append(f"  ✗ {item}")

        lines.extend(
            [
                "",
                "Recovery:",
                f"  Required: {'yes' if self.recovery_required else 'no'}",
                f"  Completed: {'yes' if self.recovery_completed else 'no'}",
                f"  Replanning: {'yes' if self.replanning_required else 'no'}",
                "",
                "Stopping Point:",
                f"  {self.stopping_point}",
                "",
                "Failure:",
                f"  {self.failure_reason}",
                "",
                f"Terminal: {'yes' if self.terminal else 'no'}",
                f"Success: {'yes' if self.success else 'no'}",
            ]
        )

        return "\n".join(lines)


def build_agent_report(
    *,
    task_id: str,
    state: Any,
    total_reward: float = 0.0,
    last_action: str = "",
    failure_reason: str = "",
) -> AgentReport:
    solved: list[str] = []
    incomplete: list[str] = []

    capabilities = (
        ("Incident observed", getattr(state, "incident_observed", False)),
        ("Diagnosis started", getattr(state, "diagnosis_started", False)),
        ("Root cause identified", getattr(state, "root_cause_identified", False)),
        ("Branch selected", getattr(state, "branch_selected", False)),
        ("Strategy selected", getattr(state, "strategy_selected", False)),
        ("Pipeline executed", getattr(state, "pipeline_run_completed", False)),
        ("Recovery completed", getattr(state, "recovery_completed", False)),
        ("Replanning completed", getattr(state, "replan_completed", False)),
        ("Global validation completed", getattr(state, "global_validation_completed", False)),
        ("Final verification", getattr(state, "terminal", False)),
    )

    for label, value in capabilities:
        if value:
            solved.append(label)
        else:
            incomplete.append(label)

    reason = failure_reason

    if not reason:
        if getattr(state, "downstream_inconsistency", False):
            reason = "downstream_inconsistency"
        elif getattr(state, "recovery_required", False):
            reason = "recovery_required"
        elif getattr(state, "replanning_required", False):
            reason = "replanning_required"
        elif not getattr(state, "success", False):
            reason = "incomplete"

    stopping_point = last_action or "not recorded"

    return AgentReport(
        task_id=task_id,
        incident=getattr(
            getattr(state, "incident", None),
            "value",
            str(getattr(state, "incident", "")),
        ),
        status="PASSED" if getattr(state, "success", False) else "FAILED",
        actions=int(getattr(state, "action_count", 0)),
        logical_stages=int(getattr(state, "logical_stage", 0)),
        total_reward=float(total_reward),
        solved=tuple(solved),
        incomplete=tuple(incomplete),
        recovery_required=bool(
            getattr(state, "recovery_required", False)
        ),
        recovery_completed=bool(
            getattr(state, "recovery_completed", False)
        ),
        replanning_required=bool(
            getattr(state, "replanning_required", False)
        ),
        downstream_inconsistency=bool(
            getattr(state, "downstream_inconsistency", False)
        ),
        stopping_point=stopping_point,
        failure_reason=reason,
        terminal=bool(getattr(state, "terminal", False)),
        success=bool(getattr(state, "success", False)),
    )
