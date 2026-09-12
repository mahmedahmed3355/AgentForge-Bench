from cuda_kernel_optimization_v1.reward import compute_reward
from cuda_kernel_optimization_v1.state import KernelOptimizationState


def test_final_success_is_dominant():
    state = KernelOptimizationState(
        logical_stage=20,
        step_count=20,
        correctness_ok=True,
        performance_improved=True,
        finalized=True,
    )

    reward = compute_reward(
        previous_stage=19,
        state=state,
    )

    assert reward.final_success == 5.0
    assert reward.total >= reward.final_success
