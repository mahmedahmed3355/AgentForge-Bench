from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .identity import ScenarioIdentity


@dataclass(frozen=True)
class Scenario:
    """Immutable scenario definition exposed to the runtime."""

    identity: ScenarioIdentity
    public: dict[str, Any] = field(default_factory=dict)
    hidden: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ScenarioIdentity):
            raise TypeError("identity must be ScenarioIdentity")

        object.__setattr__(self, "public", dict(self.public))
        object.__setattr__(self, "hidden", dict(self.hidden))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def scenario_id(self) -> str:
        return self.identity.scenario_id

    @property
    def task_id(self) -> str:
        return self.identity.task_id

    @property
    def seed(self) -> int:
        return self.identity.seed

    def agent_view(self) -> dict[str, Any]:
        """Return only agent-visible scenario information."""

        return dict(self.public)
