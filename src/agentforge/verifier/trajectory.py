from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TrajectoryVerification:
    valid: bool
    length: int
    error: str | None = None


def verify_trajectory(trajectory: Iterable[Any]) -> TrajectoryVerification:
    try:
        steps = list(trajectory)
    except TypeError:
        return TrajectoryVerification(
            valid=False,
            length=0,
            error="trajectory must be iterable",
        )

    if not steps:
        return TrajectoryVerification(
            valid=False,
            length=0,
            error="trajectory must contain at least one step",
        )

    for index, step in enumerate(steps):
        if step is None:
            return TrajectoryVerification(
                valid=False,
                length=len(steps),
                error=f"trajectory step {index} is None",
            )

    return TrajectoryVerification(
        valid=True,
        length=len(steps),
    )
