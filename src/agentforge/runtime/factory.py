from __future__ import annotations

from typing import Any

from .registry import TaskRegistry


class EnvironmentFactory:
    def __init__(self, registry: TaskRegistry) -> None:
        self._registry = registry

    def create(self, task_id: str, **kwargs: Any) -> Any:
        registration = self._registry.get(task_id)
        return registration.factory(**kwargs)
