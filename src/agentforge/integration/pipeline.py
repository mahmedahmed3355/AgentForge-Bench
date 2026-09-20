from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntegrationResult:
    success: bool
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    metadata: dict[str, Any]


class IntegrationPipeline:
    def __init__(self, runner: Any) -> None:
        if runner is None:
            raise ValueError("runner must not be None")
        self.runner = runner

    def run(self, *args: Any, **kwargs: Any) -> IntegrationResult:
        result = self.runner(*args, **kwargs)

        if not isinstance(result, IntegrationResult):
            raise TypeError("runner must return IntegrationResult")

        return result
