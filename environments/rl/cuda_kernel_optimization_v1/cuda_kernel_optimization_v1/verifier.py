from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    correctness_passed: bool
    performance_passed: bool
    final_stage_reached: bool
    score: float
    failures: tuple[str, ...]


def verify_final_state(
    logical_stage: int,
    correctness_ok: bool,
    performance_improved: bool,
    finalized: bool,
) -> VerificationResult:
    failures: list[str] = []

    if not correctness_ok:
        failures.append("correctness_not_verified")

    if not performance_improved:
        failures.append("performance_target_not_reached")

    if logical_stage < 20:
        failures.append("logical_horizon_not_completed")

    if not finalized:
        failures.append("task_not_finalized")

    passed = len(failures) == 0

    score = 1.0 if passed else 0.0

    return VerificationResult(
        passed=passed,
        correctness_passed=correctness_ok,
        performance_passed=performance_improved,
        final_stage_reached=logical_stage >= 20,
        score=score,
        failures=tuple(failures),
    )
