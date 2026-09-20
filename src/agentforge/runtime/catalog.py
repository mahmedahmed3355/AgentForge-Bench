from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDescriptor:
    task_id: str
    domain: str
    name: str
    version: str = "1"


class TaskCatalog:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskDescriptor] = {}

    def add(self, descriptor: TaskDescriptor) -> None:
        if not descriptor.task_id:
            raise ValueError("task_id must not be empty")

        if descriptor.task_id in self._tasks:
            raise ValueError(
                f"task already exists: {descriptor.task_id}"
            )

        self._tasks[descriptor.task_id] = descriptor

    def get(self, task_id: str) -> TaskDescriptor:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task: {task_id}") from exc

    def contains(self, task_id: str) -> bool:
        return task_id in self._tasks

    def all(self) -> tuple[TaskDescriptor, ...]:
        return tuple(
            self._tasks[key]
            for key in sorted(self._tasks)
        )
