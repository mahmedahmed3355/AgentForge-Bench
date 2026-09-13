from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryStep:
    step: int
    logical_stage: int
    observation: object
    action: str
    reward: float
    info: dict[str, object]


@dataclass
class EpisodeTrajectory:
    episode_id: str
    steps: list[TrajectoryStep]

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    @property
    def length(self) -> int:
        return len(self.steps)
