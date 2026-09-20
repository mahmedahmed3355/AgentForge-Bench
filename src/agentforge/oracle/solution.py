from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .behavior import OracleBehavior
from .trajectory import OracleTrajectory


@dataclass(frozen=True)
class OracleSolution:
    """Evaluator-side solution bundle."""

    trajectory: OracleTrajectory
    behavior: OracleBehavior

    @property
    def total_reward(self) -> float:
        return self.trajectory.total_reward

    def actions(self) -> tuple[Any, ...]:
        return tuple(
            step.action
            for step in self.trajectory.steps
        )


def build_solution(
    trajectory: OracleTrajectory,
    behavior: OracleBehavior,
) -> OracleSolution:
    if not isinstance(trajectory, OracleTrajectory):
        raise TypeError(
            "trajectory must be an OracleTrajectory"
        )

    if not isinstance(behavior, OracleBehavior):
        raise TypeError(
            "behavior must be an OracleBehavior"
        )

    return OracleSolution(
        trajectory=trajectory,
        behavior=behavior,
    )
