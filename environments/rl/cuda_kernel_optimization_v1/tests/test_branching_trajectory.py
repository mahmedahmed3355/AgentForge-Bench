from cuda_kernel_optimization_v1.environment import (
    CudaKernelOptimizationEnvironment,
)


def test_memory_path_creates_delayed_recovery():
    env = CudaKernelOptimizationEnvironment(seed=10)

    env.reset()
    env.step("inspect")
    env.step("modify_memory")
    env.step("compile")

    observation, reward, terminal, info = env.step(
        "check_correctness"
    )

    assert reward < 0
    assert not terminal
    assert observation["last_failure"] == "correctness"
    assert info["failure"] == "correctness"

    observation, reward, terminal, info = env.step("recover")

    assert reward > 0
    assert observation["recovery_count"] == 1
    assert observation["last_failure"] == "none"
    assert not terminal


def test_tiling_path_has_positive_delayed_outcome():
    env = CudaKernelOptimizationEnvironment(seed=11)

    env.reset()
    env.step("inspect")
    env.step("modify_tiling")
    env.step("compile")
    env.step("check_correctness")

    observation, reward, terminal, info = env.step(
        "benchmark"
    )

    assert reward > 0
    assert observation["performance_improved"]
    assert not observation["regression_detected"]
    assert info["performance_improved"]


def test_compute_path_fails_before_correctness():
    env = CudaKernelOptimizationEnvironment(seed=12)

    env.reset()
    env.step("inspect")
    env.step("modify_compute")

    observation, reward, terminal, info = env.step(
        "compile"
    )

    assert reward < 0
    assert observation["last_failure"] == "compilation"
    assert info["failure"] == "compilation"
    assert not terminal


def test_branch_choice_is_recorded_in_trajectory():
    env = CudaKernelOptimizationEnvironment(seed=13)

    env.reset()
    env.step("inspect")
    env.step("modify_tiling")

    assert env.trajectory.action_count == 2
    assert env.trajectory.transitions[1].action == "modify_tiling"
    assert (
        env.trajectory.transitions[1].info["decision"]
        == "tiling"
    )


def test_recovery_then_replanning_is_possible():
    env = CudaKernelOptimizationEnvironment(seed=14)

    env.reset()
    env.step("inspect")
    env.step("modify_memory")
    env.step("compile")
    env.step("check_correctness")
    env.step("recover")

    observation, reward, terminal, info = env.step(
        "modify_tiling"
    )

    assert reward > 0
    assert observation["optimization_branch"] == "tiling"
    assert observation["candidate_modified"]
    assert not terminal
