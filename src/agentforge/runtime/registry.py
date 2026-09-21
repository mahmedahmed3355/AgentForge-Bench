"""Legacy runtime registry compatibility adapter.

Canonical runtime storage lives in ``task_registry.py``.

This module intentionally preserves the historical ``TaskRegistration``
surface for older callers while translating legacy registrations into
canonical ``RegisteredTask`` records.

It does NOT maintain a second task store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentforge.contracts.task import (
    ActionSpec,
    CompatibilitySpec,
    EnvironmentBinding,
    ObservationSpec,
    OracleSpec,
    RewardSpec,
    ScenarioSpec,
    SuccessSpec,
    TaskIdentity,
    TaskSpec,
    TerminationSpec,
    VerifierSpec,
)
from .task_registry import (
    RegisteredTask,
    TaskRegistry as _CanonicalTaskRegistry,
)


@dataclass(frozen=True)
class TaskRegistration:
    """Compatibility view of a canonical RegisteredTask."""

    task_id: str
    factory: Callable[..., Any]
    registered_task: RegisteredTask | None = None

    @property
    def spec(self) -> TaskSpec | None:
        if self.registered_task is None:
            return None
        return self.registered_task.spec

    def make_environment(self, *args: Any, **kwargs: Any) -> Any:
        return self.factory(*args, **kwargs)


def _legacy_task_spec(task_id: str) -> TaskSpec:
    """Build the minimal declarative spec required by the canonical model.

    This exists only for legacy callers that historically registered a raw
    environment factory without a TaskSpec.

    New tasks must provide a real TaskSpec through the canonical registry.
    """

    return TaskSpec(
        identity=TaskIdentity(
            task_id=task_id,
            version="0.0.0-legacy",
            domain="legacy",
            split="legacy",
        ),
        environment=EnvironmentBinding(
            environment_id=f"legacy:{task_id}",
            environment_version="0.0.0",
            version="0.0.0",
            max_episode_steps=1,
        ),
        scenario=ScenarioSpec(
            scenario_id=f"legacy:{task_id}:scenario",
        ),
        actions=ActionSpec(
            vocabulary=("legacy",),
        ),
        observations=ObservationSpec(
            schema={"type": "legacy"},
        ),
        reward=RewardSpec(
            kind="legacy",
        ),
        termination=TerminationSpec(
            max_episode_steps=1,
        ),
        success=SuccessSpec(
            criteria=("legacy",),
        ),
        oracle=OracleSpec(
            oracle_id=f"legacy:{task_id}:oracle",
        ),
        verifier=VerifierSpec(
            verifier_id=f"legacy:{task_id}:verifier",
        ),
        compatibility=CompatibilitySpec(
            contract_version="legacy",
        ),
    )


class TaskRegistry(_CanonicalTaskRegistry):
    """Compatibility facade backed by the canonical TaskRegistry.

    The canonical registry remains the only storage owner.
    """

    def register(
        self,
        task_or_id: RegisteredTask | str,
        factory: Callable[..., Any] | None = None,
    ) -> TaskRegistration | None:
        # ---------------------------------------------------------------
        # Canonical API:
        #
        #     register(RegisteredTask(...))
        # ---------------------------------------------------------------
        if isinstance(task_or_id, RegisteredTask):
            if factory is not None:
                raise TypeError(
                    "factory must not be supplied when registering "
                    "a RegisteredTask"
                )

            return super().register(task_or_id)

        # ---------------------------------------------------------------
        # Legacy API:
        #
        #     register("task-id", factory)
        #
        # Translate it into the canonical runtime representation.
        # ---------------------------------------------------------------
        if not isinstance(task_or_id, str):
            raise TypeError(
                "register() expects RegisteredTask or legacy task_id"
            )

        if factory is None:
            raise TypeError(
                "legacy register() requires factory"
            )

        if not callable(factory):
            raise TypeError(
                "factory must be callable"
            )

        spec = _legacy_task_spec(task_or_id)

        registered = RegisteredTask(
            spec=spec,
            environment_factory=factory,
        )

        super().register(registered)

        return TaskRegistration(
            task_id=task_or_id,
            factory=factory,
            registered_task=registered,
        )

    def get(self, task_id: str) -> TaskRegistration:
        """Return the historical compatibility view.

        Canonical storage is still queried through the superclass.
        """

        registered = super().get(task_id)

        return TaskRegistration(
            task_id=registered.task_id,
            factory=registered.environment_factory,
            registered_task=registered,
        )

    def task_ids(self) -> tuple[str, ...]:
        """Historical alias preserved for compatibility."""
        return self.ids()
