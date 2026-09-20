from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestrationContext:
    """Immutable-by-convention context shared across an evaluation run."""

    task_id: str
    scenario_id: str
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **updates: Any) -> "OrchestrationContext":
        merged = dict(self.metadata)
        merged.update(updates)
        return OrchestrationContext(
            task_id=self.task_id,
            scenario_id=self.scenario_id,
            seed=self.seed,
            metadata=merged,
        )
