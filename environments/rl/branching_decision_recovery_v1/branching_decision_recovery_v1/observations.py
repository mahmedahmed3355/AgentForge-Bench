from __future__ import annotations

from .inspect_adapter import (
    inspect_system,
    inspect_component,
    inspect_dependency,
    inspect_history,
    probe_state,
    query_validation,
)

from .inspect_adapter import (
    inspect_system,
    inspect_component,
    inspect_dependency,
    inspect_history,
    probe_state,
    query_validation,
)


from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    step: int
    logical_stage: int
    system_status: str
    current_component: str | None
    visible_components: tuple[str, ...]
    available_actions: tuple[str, ...]
    observed_information: tuple[str, ...]
    warnings: tuple[str, ...]
    local_metrics: dict[str, Any]
    hypothesis: str | None
    hypothesis_confidence: float
    selected_branch: str | None
    selected_strategy: str | None
    failure_detected: bool
    recovery_required: bool
    replanning_required: bool
    terminal: bool
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "logical_stage": self.logical_stage,
            "system_status": self.system_status,
            "current_component": self.current_component,
            "visible_components": list(self.visible_components),
            "available_actions": list(self.available_actions),
            "observed_information": list(self.observed_information),
            "warnings": list(self.warnings),
            "local_metrics": dict(self.local_metrics),
            "hypothesis": self.hypothesis,
            "hypothesis_confidence": self.hypothesis_confidence,
            "selected_branch": self.selected_branch,
            "selected_strategy": self.selected_strategy,
            "failure_detected": self.failure_detected,
            "recovery_required": self.recovery_required,
            "replanning_required": self.replanning_required,
            "terminal": self.terminal,
            "success": self.success,
        }


HIDDEN_FIELDS = frozenset(
    {
        "hidden_information",
        "root_cause",
        "correct_branch",
        "optimal_branch",
        "optimal_strategy",
        "reference_solution",
        "oracle_answer",
        "ground_truth",
        "secret_solution",
    }
)


def validate_observation_contract(observation: Observation) -> None:
    payload = observation.to_dict()

    leaked = HIDDEN_FIELDS.intersection(payload)
    if leaked:
        raise AssertionError(f"Observation leaks hidden fields: {sorted(leaked)}")

    if "hidden_information" in str(payload).lower():
        raise AssertionError("Observation must not expose hidden_information.")

    if "ground_truth" in str(payload).lower():
        raise AssertionError("Observation must not expose ground_truth.")

    if "oracle_answer" in str(payload).lower():
        raise AssertionError("Observation must not expose oracle_answer.")
