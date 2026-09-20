"""Task contract."""

from dataclasses import dataclass
from typing import Any, Protocol


class TaskEnvironment(Protocol):
    """Minimal environment protocol required by a task."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class TaskContract:
    """Stable identity and version contract for benchmark tasks."""

    task_id: str
    version: str
    max_episode_steps: int

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty.")

        if not self.version:
            raise ValueError("version must not be empty.")

        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
