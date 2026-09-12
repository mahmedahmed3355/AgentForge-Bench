from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineObservation:
    logical_stage: int
    action_count: int

    incident_observed: bool
    diagnosis_started: bool
    root_cause_identified: bool

    selected_branch: Optional[str]
    selected_strategy: Optional[str]

    schema_valid: bool
    cleaning_valid: bool
    transformation_valid: bool
    aggregation_valid: bool
    output_valid: bool

    pipeline_run_completed: bool
    failure_detected: bool
    recovery_required: bool
    recovery_completed: bool
    replanning_required: bool

    downstream_inconsistency: bool
    global_validation_completed: bool

    terminal: bool
    success: bool

    @classmethod
    def from_state(cls, state):
        return cls(
            logical_stage=state.logical_stage,
            action_count=state.action_count,
            incident_observed=state.incident_observed,
            diagnosis_started=state.diagnosis_started,
            root_cause_identified=state.root_cause_identified,
            selected_branch=state.selected_branch,
            selected_strategy=state.selected_strategy,
            schema_valid=state.schema_valid,
            cleaning_valid=state.cleaning_valid,
            transformation_valid=state.transformation_valid,
            aggregation_valid=state.aggregation_valid,
            output_valid=state.output_valid,
            pipeline_run_completed=state.pipeline_run_completed,
            failure_detected=state.failure_detected,
            recovery_required=state.recovery_required,
            recovery_completed=state.recovery_completed,
            replanning_required=state.replanning_required,
            downstream_inconsistency=state.downstream_inconsistency,
            global_validation_completed=state.global_validation_completed,
            terminal=state.terminal,
            success=state.success,
        )
