from __future__ import annotations

from dataclasses import dataclass

from .causal import PipelineComponent
from .decision_graph import IncidentKind, RecoveryStrategy


@dataclass(frozen=True)
class OracleResult:
    success: bool
    expected_branch: str
    expected_strategy: str
    minimum_logical_stage: int
    reason: str


_EXPECTED = {
    IncidentKind.SCHEMA: (
        "schema_recovery",
        RecoveryStrategy.REPAIR_UPSTREAM,
        14,
    ),
    IncidentKind.TRANSFORMATION: (
        "transformation_recovery",
        RecoveryStrategy.PATCH_TRANSFORM,
        16,
    ),
    IncidentKind.DATA_QUALITY: (
        "data_quality_recovery",
        RecoveryStrategy.FILTER_INVALID,
        17,
    ),
    IncidentKind.RESOURCE: (
        "resource_recovery",
        RecoveryStrategy.OPTIMIZE_EXECUTION,
        17,
    ),
}


class PipelineRecoveryOracle:
    def expected(self, incident: IncidentKind) -> OracleResult:
        branch, strategy, minimum_stage = _EXPECTED[incident]
        return OracleResult(
            success=True,
            expected_branch=branch,
            expected_strategy=strategy.value,
            minimum_logical_stage=minimum_stage,
            reason=f"reference recovery policy for {incident.value}",
        )

    def evaluate(
        self,
        incident: IncidentKind,
        *,
        branch: str | None,
        strategy: RecoveryStrategy | None,
        logical_stage: int,
        all_components_valid: bool,
        output_valid: bool,
        recovery_required: bool,
        replanning_required: bool,
        downstream_inconsistency: bool,
        pipeline_run_completed: bool,
    ) -> OracleResult:
        reference = self.expected(incident)

        valid = (
            logical_stage >= reference.minimum_logical_stage
            and branch == reference.expected_branch
            and strategy is not None
            and strategy.value == reference.expected_strategy
            and all_components_valid
            and output_valid
            and not recovery_required
            and not replanning_required
            and not downstream_inconsistency
            and pipeline_run_completed
        )

        return OracleResult(
            success=valid,
            expected_branch=reference.expected_branch,
            expected_strategy=reference.expected_strategy,
            minimum_logical_stage=reference.minimum_logical_stage,
            reason=(
                "reference trajectory satisfies all terminal invariants"
                if valid
                else "reference terminal invariants not satisfied"
            ),
        )


def expected_recovery_policy(incident: IncidentKind) -> tuple[str, RecoveryStrategy, int]:
    return _EXPECTED[incident]
