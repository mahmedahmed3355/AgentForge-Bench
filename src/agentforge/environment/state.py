from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentState:
    episode_id: str
    step_count: int = 0
    terminated: bool = False
    truncated: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")

        if self.step_count < 0:
            raise ValueError("step_count must be non-negative")

        if self.terminated and self.truncated:
            raise ValueError(
                "terminated and truncated cannot both be True"
            )

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def mark_terminated(self) -> None:
        if self.truncated:
            raise ValueError(
                "cannot terminate an already truncated episode"
            )
        self.terminated = True

    def mark_truncated(self) -> None:
        if self.terminated:
            raise ValueError(
                "cannot truncate an already terminated episode"
            )
        self.truncated = True
