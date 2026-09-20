from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardVerification:
    valid: bool
    value: float
    error: str | None = None


def verify_reward(value: float) -> RewardVerification:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return RewardVerification(
            valid=False,
            value=0.0,
            error="reward must be numeric",
        )

    if not math.isfinite(numeric):
        return RewardVerification(
            valid=False,
            value=numeric,
            error="reward must be finite",
        )

    return RewardVerification(
        valid=True,
        value=numeric,
    )
