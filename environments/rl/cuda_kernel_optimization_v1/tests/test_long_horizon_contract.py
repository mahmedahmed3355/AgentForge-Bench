from cuda_kernel_optimization_v1.environment import (
    CudaKernelOptimizationEnvironment,
)


def test_success_path_requires_logical_horizon_completion():
    env = CudaKernelOptimizationEnvironment(seed=21)

    env.reset()
    env.step("inspect")
    env.step("modify_tiling")
    env.step("compile")
    env.step("check_correctness")
    env.step("benchmark")
    env.step("analyze")

    assert env.state.logical_stage == 15
    assert env.state.action_count == 6

    observation, reward, terminal, info = env.step(
        "final_verify"
    )

    assert reward < 0
    assert not terminal
    assert not observation["success"]


def test_logical_stage_and_action_count_are_independent():
    env = CudaKernelOptimizationEnvironment(seed=22)

    env.reset()

    for action in [
        "inspect",
        "modify_tiling",
        "compile",
        "check_correctness",
        "benchmark",
        "analyze",
    ]:
        env.step(action)

    assert env.state.action_count == 6
    assert env.state.logical_stage == 15
    assert env.state.logical_stage != env.state.action_count
