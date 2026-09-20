from __future__ import annotations

from typing import Any


class AgentForgeBenchAdapter:
    """Stable benchmark-facing adapter boundary."""

    def __init__(self, environment: Any) -> None:
        self.environment = environment

    def reset(self, *, seed: int | None = None) -> Any:
        return self.environment.reset(seed=seed)

    def step(self, action: Any) -> Any:
        return self.environment.step(action)

    def close(self) -> None:
        self.environment.close()
