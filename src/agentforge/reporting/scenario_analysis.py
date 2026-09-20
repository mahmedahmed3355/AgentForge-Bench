from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScenarioAnalysis:
    scenario_ids: Sequence[str]

    @property
    def count(self) -> int:
        return len(self.scenario_ids)

    @property
    def unique_count(self) -> int:
        return len(set(self.scenario_ids))
