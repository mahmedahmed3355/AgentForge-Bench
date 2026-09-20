from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskResultReport:
    task_id: str
    success: bool
    reward: float
    episodes: int
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reward": self.reward,
            "episodes": self.episodes,
            "details": dict(self.details),
        }
