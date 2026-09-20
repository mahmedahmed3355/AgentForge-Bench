from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneralizationReport:
    seen_successes: int
    seen_total: int
    unseen_successes: int
    unseen_total: int

    @property
    def seen_rate(self) -> float:
        if self.seen_total == 0:
            return 0.0
        return self.seen_successes / self.seen_total

    @property
    def unseen_rate(self) -> float:
        if self.unseen_total == 0:
            return 0.0
        return self.unseen_successes / self.unseen_total
