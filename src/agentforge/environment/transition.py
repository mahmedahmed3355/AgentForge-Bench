from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.reward, (int, float)):
            raise TypeError("reward must be numeric")

        if self.reward != self.reward:
            raise ValueError("reward must be a finite scalar")

        if self.reward in (float("inf"), float("-inf")):
            raise ValueError("reward must be a finite scalar")

        if self.terminated and self.truncated:
            raise ValueError(
                "terminated and truncated cannot both be True"
            )

        if not isinstance(self.info, dict):
            raise TypeError("info must be a dictionary")
