from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuccessVerification:
    success: bool
    reason: str


def verify_success(
    *,
    episode_valid: bool,
    reward_valid: bool,
    trajectory_valid: bool,
) -> SuccessVerification:
    if not episode_valid:
        return SuccessVerification(
            success=False,
            reason="episode verification failed",
        )

    if not reward_valid:
        return SuccessVerification(
            success=False,
            reason="reward verification failed",
        )

    if not trajectory_valid:
        return SuccessVerification(
            success=False,
            reason="trajectory verification failed",
        )

    return SuccessVerification(
        success=True,
        reason="all verifier checks passed",
    )
