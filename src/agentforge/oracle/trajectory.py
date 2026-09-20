from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OracleStep:
    """One evaluator-side reference action and resulting observation."""

    action: Any
    observation: Any = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        value = float(self.reward)

        if value != value:
            raise ValueError("oracle reward must be finite")

        if value in (float("inf"), float("-inf")):
            raise ValueError("oracle reward must be finite")


@dataclass(frozen=True)
class OracleTrajectory:
    """Ordered evaluator-side reference trajectory."""

    steps: tuple[OracleStep, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)

    @property
    def terminated(self) -> bool:
        return bool(self.steps) and self.steps[-1].terminated

    @property
    def truncated(self) -> bool:
        return bool(self.steps) and self.steps[-1].truncated

    def append(self, step: OracleStep) -> "OracleTrajectory":
        if not isinstance(step, OracleStep):
            raise TypeError("step must be an OracleStep")

        return OracleTrajectory(
            steps=self.steps + (step,),
        )
