from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from .base import Scenario
from .identity import ScenarioIdentity


class ScenarioGenerator(ABC):
    """Base class for deterministic scenario generation."""

    def __init__(self, task_id: str) -> None:
        if not task_id:
            raise ValueError("task_id must be non-empty")
        self.task_id = task_id

    @abstractmethod
    def generate_data(self, rng: random.Random) -> dict[str, Any]:
        """Generate scenario data using the supplied local RNG."""

    def generate(
        self,
        *,
        scenario_id: str,
        seed: int,
    ) -> Scenario:
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        rng = random.Random(seed)
        data = self.generate_data(rng)

        if not isinstance(data, dict):
            raise TypeError("generate_data() must return a dictionary")

        public = dict(data.get("public", {}))
        hidden = dict(data.get("hidden", {}))
        metadata = dict(data.get("metadata", {}))

        return Scenario(
            identity=ScenarioIdentity(
                task_id=self.task_id,
                scenario_id=scenario_id,
                seed=seed,
            ),
            public=public,
            hidden=hidden,
            metadata=metadata,
        )


class DemoScenarioGenerator(ScenarioGenerator):
    """Small framework-level generator used by contract tests."""

    def generate_data(self, rng: random.Random) -> dict[str, Any]:
        return {
            "public": {
                "value": rng.randint(0, 100),
            },
            "hidden": {
                "target": rng.randint(0, 100),
            },
            "metadata": {
                "generator": "demo",
            },
        }
