from cuda_kernel_optimization_v1.decision_graph import (
    DecisionKind,
    get_branch_outcome,
)


def test_all_branches_have_distinct_outcomes():
    outcomes = [
        get_branch_outcome(DecisionKind.MEMORY),
        get_branch_outcome(DecisionKind.TILING),
        get_branch_outcome(DecisionKind.COMPUTE),
    ]

    assert len(outcomes) == 3
    assert outcomes[0].correctness_success is False
    assert outcomes[1].correctness_success is True
    assert outcomes[2].compile_success is False


def test_tiling_is_the_success_branch():
    outcome = get_branch_outcome(DecisionKind.TILING)

    assert outcome.compile_success
    assert outcome.correctness_success
    assert outcome.benchmark_success
    assert not outcome.recovery_required


def test_non_tiling_branches_require_recovery():
    assert get_branch_outcome(DecisionKind.MEMORY).recovery_required
    assert get_branch_outcome(DecisionKind.COMPUTE).recovery_required
