from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardComponent:
    """One named contribution to an episode reward."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reward component name must not be empty")

        value = float(self.value)

        if value != value:
            raise ValueError("reward component must be finite")

        if value in (float("inf"), float("-inf")):
            raise ValueError("reward component must be finite")
