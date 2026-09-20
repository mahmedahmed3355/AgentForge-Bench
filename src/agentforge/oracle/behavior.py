from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .trajectory import OracleTrajectory


@dataclass(frozen=True)
class OracleBehavior:
    """Evaluator-side callable reference behavior."""

    name: str
    policy: Callable[[Any], Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("oracle behavior name must not be empty")

        if not callable(self.policy):
            raise TypeError("oracle policy must be callable")

    def act(self, observation: Any) -> Any:
        return self.policy(observation)


def replay_behavior(
    behavior: OracleBehavior,
    observations: list[Any],
) -> list[Any]:
    if not isinstance(behavior, OracleBehavior):
        raise TypeError("behavior must be an OracleBehavior")

    return [
        behavior.act(observation)
        for observation in observations
    ]


def behavior_matches_trajectory(
    behavior: OracleBehavior,
    observations: list[Any],
    trajectory: OracleTrajectory,
) -> bool:
    actions = replay_behavior(behavior, observations)

    expected = [
        step.action
        for step in trajectory.steps
    ]

    return actions == expected
