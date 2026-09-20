from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioIdentity:
    """Stable identity for a generated scenario."""

    task_id: str
    scenario_id: str
    seed: int

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    @property
    def key(self) -> str:
        return f"{self.task_id}:{self.scenario_id}:{self.seed}"
