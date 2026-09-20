from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CapabilityReport:
    capability: str
    attempts: int
    successes: int

    @property
    def success_rate(self) -> float:
        if self.attempts < 0 or self.successes < 0:
            raise ValueError("attempts and successes must be non-negative")
        if self.successes > self.attempts:
            raise ValueError("successes cannot exceed attempts")
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @classmethod
    def from_mapping(
        cls,
        capability: str,
        values: Mapping[str, int],
    ) -> "CapabilityReport":
        return cls(
            capability=capability,
            attempts=int(values.get("attempts", 0)),
            successes=int(values.get("successes", 0)),
        )
