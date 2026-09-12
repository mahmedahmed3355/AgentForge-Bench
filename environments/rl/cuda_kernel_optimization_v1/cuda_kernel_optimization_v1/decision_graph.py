from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DecisionKind(StrEnum):
    MEMORY = "memory"
    TILING = "tiling"
    COMPUTE = "compute"


@dataclass(frozen=True)
class BranchOutcome:
    branch: DecisionKind
    compile_success: bool
    correctness_success: bool
    benchmark_success: bool
    delayed_regression: bool
    recovery_required: bool
    expected_stage_delta: int


BRANCH_OUTCOMES = {
    DecisionKind.MEMORY: BranchOutcome(
        branch=DecisionKind.MEMORY,
        compile_success=True,
        correctness_success=False,
        benchmark_success=False,
        delayed_regression=False,
        recovery_required=True,
        expected_stage_delta=6,
    ),
    DecisionKind.TILING: BranchOutcome(
        branch=DecisionKind.TILING,
        compile_success=True,
        correctness_success=True,
        benchmark_success=True,
        delayed_regression=False,
        recovery_required=False,
        expected_stage_delta=13,
    ),
    DecisionKind.COMPUTE: BranchOutcome(
        branch=DecisionKind.COMPUTE,
        compile_success=False,
        correctness_success=False,
        benchmark_success=False,
        delayed_regression=True,
        recovery_required=True,
        expected_stage_delta=3,
    ),
}


def get_branch_outcome(branch: DecisionKind) -> BranchOutcome:
    return BRANCH_OUTCOMES[branch]
