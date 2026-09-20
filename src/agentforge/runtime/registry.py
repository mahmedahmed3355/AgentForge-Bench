from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TaskRegistration:
    task_id: str
    factory: Callable[..., Any]


class TaskRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, TaskRegistration] = {}

    def register(
        self,
        task_id: str,
        factory: Callable[..., Any],
    ) -> TaskRegistration:
        if not task_id:
            raise ValueError("task_id must not be empty")

        if task_id in self._registrations:
            raise ValueError(f"task already registered: {task_id}")

        registration = TaskRegistration(
            task_id=task_id,
            factory=factory,
        )
        self._registrations[task_id] = registration
        return registration

    def get(self, task_id: str) -> TaskRegistration:
        try:
            return self._registrations[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task: {task_id}") from exc

    def contains(self, task_id: str) -> bool:
        return task_id in self._registrations

    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))
