from cuda_kernel_optimization_v1.requirements import (
    CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS,
    validate_requirements,
)


def test_cuda_task_satisfies_long_horizon_contract():
    validate_requirements()


def test_cuda_task_has_20_logical_stages():
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.logical_stages == 20


def test_cuda_task_has_branching_and_recovery():
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.branching
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.decision_making
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.recovery
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.delayed_consequences


def test_cuda_task_has_evaluation_isolation():
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.hidden_evaluation
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.unseen_scenarios
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.independent_verifier


def test_cuda_task_is_prime_and_rl_ready():
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.prime_v1_compatible
    assert CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.rl_training_ready
