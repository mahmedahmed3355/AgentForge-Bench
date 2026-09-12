from __future__ import annotations

from dataclasses import dataclass

from .state import KernelOptimizationState


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    progress: float
    milestone: float
    recovery: float
    correctness: float
    performance: float
    efficiency: float
    final_success: float


def compute_reward(
    previous_stage: int,
    state: KernelOptimizationState,
) -> RewardBreakdown:
    progress = max(0.0, state.logical_stage - previous_stage) * 0.10
    milestone = 0.0

    if state.logical_stage in {5, 10, 15, 20}:
        milestone = 0.25

    recovery = 0.20 if state.recovery_count > 0 else 0.0
    correctness = 0.50 if state.correctness_ok else 0.0
    performance = 0.75 if state.performance_improved else 0.0

    efficiency = max(
        0.0,
        0.10 - (0.01 * max(0, state.step_count - state.logical_stage)),
    )

    final_success = 5.0 if state.finalized and state.correctness_ok and state.performance_improved else 0.0

    total = (
        progress
        + milestone
        + recovery
        + correctness
        + performance
        + efficiency
        + final_success
    )

    return RewardBreakdown(
        total=total,
        progress=progress,
        milestone=milestone,
        recovery=recovery,
        correctness=correctness,
        performance=performance,
        efficiency=efficiency,
        final_success=final_success,
    )
