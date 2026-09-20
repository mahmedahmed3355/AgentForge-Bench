from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeVerification:
    valid: bool
    step_count: int
    terminated: bool
    truncated: bool
    error: str | None = None


def verify_episode(
    *,
    step_count: int,
    terminated: bool,
    truncated: bool,
) -> EpisodeVerification:
    if step_count < 0:
        return EpisodeVerification(
            valid=False,
            step_count=step_count,
            terminated=terminated,
            truncated=truncated,
            error="step_count must be non-negative",
        )

    if terminated and truncated:
        return EpisodeVerification(
            valid=False,
            step_count=step_count,
            terminated=terminated,
            truncated=truncated,
            error="terminated and truncated cannot both be true",
        )

    if not terminated and not truncated:
        return EpisodeVerification(
            valid=False,
            step_count=step_count,
            terminated=terminated,
            truncated=truncated,
            error="episode must end with termination or truncation",
        )

    return EpisodeVerification(
        valid=True,
        step_count=step_count,
        terminated=terminated,
        truncated=truncated,
    )
