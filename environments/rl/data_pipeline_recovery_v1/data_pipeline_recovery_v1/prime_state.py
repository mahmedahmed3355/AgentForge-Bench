from __future__ import annotations

from pydantic import Field
from verifiers.v1.state import State


class PipelinePrimeState(State):
    """Prime V1 transport state for Data Pipeline Recovery V1."""

    seed: int | None = None
    incident: str = "schema"

    logical_stage: int = 0
    action_count: int = 0

    incident_observed: bool = False
    diagnosis_started: bool = False
    root_cause_identified: bool = False

    selected_branch: str | None = None
    selected_strategy: str | None = None
    recovery_mode: str | None = None

    inspected_components: list[str] = Field(default_factory=list)

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

    last_observation: str = ""
    history: list[str] = Field(default_factory=list)
