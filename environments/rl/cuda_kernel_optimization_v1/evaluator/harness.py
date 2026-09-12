from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cuda_kernel_optimization_v1.oracle import solve_oracle
from cuda_kernel_optimization_v1.verifier import verify_final_state
from cuda_kernel_optimization_v1.environment import (
    CudaKernelOptimizationEnvironment,
)


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    scenario_id: str
    success: bool
    score: float
    verifier_passed: bool
    oracle_passed: bool
    final_stage: int
    correctness_passed: bool
    performance_passed: bool
    action_count: int
    recovery_count: int
    failures: tuple[str, ...]


class EvaluatorHarness:
    """
    Evaluator-only boundary for CUDA Kernel Optimization V1.

    The agent interacts only with the public environment.

    The evaluator owns:
    - hidden scenario selection
    - final verification
    - oracle/reference verification

    Hidden scenario data is never exposed through the public taskset.
    """

    def __init__(self) -> None:
        self._oracle_result = solve_oracle()

    def evaluate(
        self,
        *,
        task_id: str,
        scenario_id: str,
        seed: int,
        actions: list[str],
    ) -> EvaluationResult:
        env = CudaKernelOptimizationEnvironment(seed=seed)
        env.reset(seed=seed)

        for action in actions:
            if env.state.terminal:
                break
            env.step(action)

        state = env.state

        verifier_result = verify_final_state(
            logical_stage=state.logical_stage,
            correctness_ok=state.correctness_ok,
            performance_improved=state.performance_improved,
            finalized=state.finalized,
        )

        oracle_result = self._oracle_result

        failures: list[str] = []

        if not verifier_result.passed:
            failures.extend(verifier_result.failures)

        if not oracle_result.passed:
            failures.extend(oracle_result.failures)

        return EvaluationResult(
            task_id=task_id,
            scenario_id=scenario_id,
            success=bool(
                verifier_result.passed
                and oracle_result.passed
            ),
            score=min(
                float(verifier_result.score),
                float(oracle_result.score),
            ),
            verifier_passed=bool(verifier_result.passed),
            oracle_passed=bool(oracle_result.passed),
            final_stage=state.logical_stage,
            correctness_passed=state.correctness_ok,
            performance_passed=state.performance_improved,
            action_count=state.action_count,
            recovery_count=state.recovery_count,
            failures=tuple(dict.fromkeys(failures)),
        )

    @staticmethod
    def result_to_dict(
        result: EvaluationResult,
    ) -> dict[str, Any]:
        return {
            "task_id": result.task_id,
            "scenario_id": result.scenario_id,
            "success": result.success,
            "score": result.score,
            "verifier_passed": result.verifier_passed,
            "oracle_passed": result.oracle_passed,
            "final_stage": result.final_stage,
            "correctness_passed": result.correctness_passed,
            "performance_passed": result.performance_passed,
            "action_count": result.action_count,
            "recovery_count": result.recovery_count,
            "failures": list(result.failures),
        }
