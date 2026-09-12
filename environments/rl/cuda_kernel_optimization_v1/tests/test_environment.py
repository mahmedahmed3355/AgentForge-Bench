from cuda_kernel_optimization_v1.environment import (
    CudaKernelOptimizationEnvironment,
    FailureType,
    OptimizationBranch,
)


def test_environment_reset_is_deterministic():
    env = CudaKernelOptimizationEnvironment(seed=123)

    first = env.reset(seed=123)
    second = env.reset(seed=123)

    assert first == second


def test_branch_choice_changes_state():
    env = CudaKernelOptimizationEnvironment(seed=1)

    env.reset()
    env.step("inspect")
    env.step("modify_tiling")

    assert env.state.optimization_branch == OptimizationBranch.TILING
    assert env.state.candidate_modified


def test_different_branch_has_different_consequence():
    tiling = CudaKernelOptimizationEnvironment(seed=1)
    tiling.reset()
    tiling.step("inspect")
    tiling.step("modify_tiling")
    tiling.step("compile")
    tiling.step("check_correctness")
    tiling.step("benchmark")

    compute = CudaKernelOptimizationEnvironment(seed=1)
    compute.reset()
    compute.step("inspect")
    compute.step("modify_compute")
    compute.step("compile")

    assert tiling.state.performance_improved
    assert tiling.state.regression_detected is False
    assert compute.state.last_failure == FailureType.COMPILATION


def test_memory_branch_requires_recovery():
    env = CudaKernelOptimizationEnvironment(seed=2)

    env.reset()
    env.step("inspect")
    env.step("modify_memory")
    env.step("compile")

    _, reward, _, _ = env.step("check_correctness")

    assert reward < 0
    assert env.state.last_failure == FailureType.CORRECTNESS

    _, recovery_reward, _, _ = env.step("recover")

    assert recovery_reward > 0
    assert env.state.recovery_count == 1
    assert env.state.last_failure == FailureType.NONE


def test_tiling_branch_reaches_success_path():
    env = CudaKernelOptimizationEnvironment(seed=3)

    env.reset()
    env.step("inspect")
    env.step("modify_tiling")
    env.step("compile")
    env.step("check_correctness")
    env.step("benchmark")
    env.step("analyze")

    assert env.state.performance_improved
    assert env.state.correctness_ok
    assert env.state.analysis_completed


def test_final_verification_requires_success_conditions():
    env = CudaKernelOptimizationEnvironment(seed=4)

    env.reset()
    _, reward, terminal, _ = env.step("final_verify")

    assert reward < 0
    assert not terminal
    assert not env.state.success


def test_logical_stage_is_not_action_count():
    env = CudaKernelOptimizationEnvironment(seed=5)

    env.reset()

    env.step("inspect")
    env.step("modify_tiling")
    env.step("compile")
    env.step("check_correctness")
    env.step("benchmark")
    env.step("analyze")

    assert env.state.action_count == 6
    assert env.state.logical_stage > 6
