from cuda_kernel_optimization_v1.oracle import solve_oracle


def test_oracle_passes():
    result = solve_oracle()

    assert result.passed
    assert result.correctness_passed
    assert result.performance_passed
    assert result.final_stage_reached
