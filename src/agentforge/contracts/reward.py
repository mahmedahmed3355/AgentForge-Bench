"""Reward contract."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardContract:
    """Canonical scalar finite reward contract."""

    allow_negative: bool = True

    def validate(self, reward: float) -> float:
        if isinstance(reward, bool):
            raise ValueError("Reward must be a numeric scalar.")

        value = float(reward)

        if not math.isfinite(value):
            raise ValueError("Reward must be finite.")

        if not self.allow_negative and value < 0:
            raise ValueError("Negative reward is not allowed.")

        return value
