from cuda_kernel_optimization_v1.environment import CudaKernelOptimizationEnvironment
from cuda_kernel_optimization_v1.requirements import (
    CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS,
)


def test_environment_matches_task_contract():
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.logical_stages == 20
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.stateful
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.branching
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.decision_making
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.delayed_consequences
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.recovery


def test_environment_is_stateful():
    env = CudaKernelOptimizationEnvironment(seed=99)

    initial = env.reset()
    after_inspect, _, _, _ = env.step("inspect")

    assert initial["logical_stage"] == 0
    assert after_inspect["logical_stage"] == 2
    assert after_inspect["inspected"]
