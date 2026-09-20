from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SummaryReport:
    total_tasks: int
    successful_tasks: int
    total_reward: float

    @property
    def task_success_rate(self) -> float:
        if self.total_tasks < 0 or self.successful_tasks < 0:
            raise ValueError("task counts must be non-negative")
        if self.successful_tasks > self.total_tasks:
            raise ValueError(
                "successful_tasks cannot exceed total_tasks"
            )
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "total_reward": self.total_reward,
            "task_success_rate": self.task_success_rate,
        }
