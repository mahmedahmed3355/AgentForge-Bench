from __future__ import annotations

from dataclasses import dataclass, field

from .decision_graph import (
    IncidentKind,
    RecoveryMode,
    RecoveryStrategy,
)


PIPELINE_COMPONENTS = (
    "input",
    "schema_validation",
    "cleaning",
    "transformation",
    "aggregation",
    "output_validation",
    "output",
)


@dataclass
class PipelineState:
    episode_id: str
    incident: IncidentKind = IncidentKind.SCHEMA

    logical_stage: int = 0
    action_count: int = 0

    incident_observed: bool = False
    diagnosis_started: bool = False
    root_cause_identified: bool = False

    selected_branch: str | None = None
    selected_strategy: RecoveryStrategy | str | None = None
    recovery_mode: RecoveryMode | None = None

    inspected_components: set[str] = field(default_factory=set)

    component_validity: dict[str, bool] = field(
        default_factory=lambda: {
            "input": True,
            "schema_validation": False,
            "cleaning": False,
            "transformation": False,
            "aggregation": False,
            "output_validation": False,
            "output": False,
        }
    )

    blocked_components: set[str] = field(default_factory=set)
    downstream_invalidated: set[str] = field(default_factory=set)

    schema_valid: bool = False
    cleaning_valid: bool = False
    transformation_valid: bool = False
    aggregation_valid: bool = False
    output_valid: bool = False

    pipeline_run_completed: bool = False
    failure_detected: bool = False

    recovery_required: bool = False
    recovery_completed: bool = False

    replanning_required: bool = False
    replan_completed: bool = False

    downstream_inconsistency: bool = False

    strategy_validated: bool = False
    global_validation_completed: bool = False

    terminal: bool = False
    success: bool = False

    @property
    def all_components_valid(self) -> bool:
        return (
            self.schema_valid
            and self.cleaning_valid
            and self.transformation_valid
            and self.aggregation_valid
            and self.output_valid
        )

    @property
    def branch_selected(self) -> bool:
        return self.selected_branch is not None

    @property
    def strategy_selected(self) -> bool:
        return self.selected_strategy is not None
