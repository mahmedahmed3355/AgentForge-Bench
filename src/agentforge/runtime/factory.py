"""Canonical environment factory."""

from __future__ import annotations

from typing import Any

from .task_registry import TaskRegistry


class EnvironmentFactory:
    """Create agent-facing environments from the canonical task registry."""

    def __init__(self, registry: TaskRegistry) -> None:
        self._registry = registry

    def create(self, task_id: str, **kwargs: Any) -> Any:
        registration = self._registry.get(task_id)

        # Canonical RegisteredTask.
        if hasattr(registration, "make_environment"):
            return registration.make_environment(**kwargs)

        # Compatibility TaskRegistration.
        if hasattr(registration, "factory"):
            return registration.factory(**kwargs)

        raise TypeError(
            "registered task does not expose an environment factory"
        )
