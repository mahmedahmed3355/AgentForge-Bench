from __future__ import annotations
"""Canonical AgentForge task registry.

The registry is responsible for binding a TaskSpec to the runtime
components required to execute and evaluate that task.

Canonical runtime components:
    RegisteredTask
    ├── spec
    ├── environment_factory
    ├── scenario_factory
    ├── reward_factory
    ├── oracle
    └── verifier

The oracle and verifier are evaluator-only components. They are stored
on the registration record so the evaluator can access them, but they
are not used by the agent-facing environment execution path."""

from dataclasses import dataclass
from typing import Any, Callable
from agentforge.contracts.task import TaskSpec

EnvironmentFactory = Callable[..., Any]
ScenarioFactory = Callable[..., Any]
RewardFactory = Callable[..., Any]
Oracle = Any
Verifier = Any


@dataclass(frozen=True)
class RegisteredTask:
    """Complete runtime binding for one benchmark task."""

    spec: TaskSpec
    environment_factory: EnvironmentFactory
    scenario_factory: ScenarioFactory | None = None
    reward_factory: RewardFactory | None = None
    oracle: Oracle | None = None
    verifier: Verifier | None = None

    def __post_init__(self) -> None:
        if self.spec is None:
            raise ValueError("spec must not be None")
        if not callable(self.environment_factory):
            raise TypeError("environment_factory must be callable")
        if self.scenario_factory is not None:
            if not callable(self.scenario_factory):
                raise TypeError("scenario_factory must be callable")
        if self.reward_factory is not None:
            if not callable(self.reward_factory):
                raise TypeError("reward_factory must be callable")

    @property
    def task_id(self) -> str:
        return self.spec.identity.task_id

    @property
    def version(self) -> str:
        return self.spec.identity.version

    @property
    def has_evaluator(self) -> bool:
        return self.oracle is not None or self.verifier is not None

    @property
    def has_oracle(self) -> bool:
        return self.oracle is not None

    @property
    def has_verifier(self) -> bool:
        return self.verifier is not None

    def make_environment(self, *args: Any, **kwargs: Any) -> Any:
        """Create the agent-facing environment instance."""
        return self.environment_factory(*args, **kwargs)

    def make_scenario(self, *args: Any, **kwargs: Any) -> Any:
        """Create a scenario using the registered scenario factory."""
        if self.scenario_factory is None:
            raise RuntimeError(
                f"task {self.task_id!r} has no scenario_factory"
            )
        return self.scenario_factory(*args, **kwargs)

    def make_reward_engine(self, *args: Any, **kwargs: Any) -> Any:
        """Create the task reward engine using the registered factory."""
        if self.reward_factory is None:
            raise RuntimeError(
                f"task {self.task_id!r} has no reward_factory"
            )
        return self.reward_factory(*args, **kwargs)


class TaskRegistry:
    """Canonical registry of benchmark task runtime bindings."""

    def __init__(self) -> None:
        self._tasks: dict[str, RegisteredTask] = {}

    def register(self, task: RegisteredTask) -> None:
        """Register one task by canonical task id."""
        task_id = task.task_id
        if task_id in self._tasks:
            raise ValueError(
                f"task already registered: {task_id!r}"
            )
        self._tasks[task_id] = task

    def replace(self, task: RegisteredTask) -> None:
        """Explicitly replace an existing task registration."""
        self._tasks[task.task_id] = task

    def unregister(self, task_id: str) -> None:
        """Remove a task registration."""
        if task_id not in self._tasks:
            raise KeyError(task_id)
        del self._tasks[task_id]

    def get(self, task_id: str) -> RegisteredTask:
        """Return a registered task or raise KeyError."""
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(
                f"task is not registered: {task_id!r}"
            ) from None

    def contains(self, task_id: str) -> bool:
        return task_id in self._tasks

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def ids(self) -> tuple[str, ...]:
        """Return registered task ids in deterministic order."""
        return tuple(sorted(self._tasks))


    def task_ids(self) -> tuple[str, ...]:
        """Compatibility alias for the legacy TaskRegistry.task_ids API."""
        return self.ids()

    def all(self) -> tuple[RegisteredTask, ...]:
        """Return all registered tasks in deterministic order."""
        return tuple(
            self._tasks[task_id]
            for task_id in sorted(self._tasks)
        )
