from __future__ import annotations

from dataclasses import dataclass

from .decision_graph import IncidentKind, RecoveryStrategy
from .oracle import PipelineRecoveryOracle
from .state import PipelineState


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    terminal: bool
    logical_stage: int
    failure_reason: str | None


class PipelineRecoveryVerifier:
    def __init__(self) -> None:
        self._oracle = PipelineRecoveryOracle()

    def verify(self, state: PipelineState) -> VerificationResult:
        if not state.terminal:
            return VerificationResult(
                success=False,
                terminal=False,
                logical_stage=state.logical_stage,
                failure_reason="episode is not terminal",
            )

        if not state.success:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="environment marked episode unsuccessful",
            )

        incident = state.incident
        reference = self._oracle.expected(incident)

        if state.selected_branch != reference.expected_branch:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="selected branch does not match reference branch",
            )

        if state.selected_strategy != reference.expected_strategy:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="selected strategy does not match reference strategy",
            )

        if state.logical_stage < reference.minimum_logical_stage:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="terminal trajectory is too short",
            )

        if not state.all_components_valid:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="not all pipeline components are valid",
            )

        if not state.output_valid:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="output validation is incomplete",
            )

        if state.recovery_required:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="recovery remains required",
            )

        if state.replanning_required:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="replanning remains required",
            )

        if state.downstream_inconsistency:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="downstream inconsistency remains",
            )

        if not state.pipeline_run_completed:
            return VerificationResult(
                success=False,
                terminal=True,
                logical_stage=state.logical_stage,
                failure_reason="pipeline was not successfully executed",
            )

        return VerificationResult(
            success=True,
            terminal=True,
            logical_stage=state.logical_stage,
            failure_reason=None,
        )
