from cuda_kernel_optimization_v1.verifier import verify_final_state


def test_oracle_final_state_passes():
    result = verify_final_state(
        logical_stage=20,
        correctness_ok=True,
        performance_improved=True,
        finalized=True,
    )

    assert result.passed
    assert result.score == 1.0
    assert not result.failures


def test_incomplete_state_fails():
    result = verify_final_state(
        logical_stage=10,
        correctness_ok=True,
        performance_improved=False,
        finalized=False,
    )

    assert not result.passed
    assert "logical_horizon_not_completed" in result.failures
    assert "performance_target_not_reached" in result.failures
