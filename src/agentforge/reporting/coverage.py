from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageReport:
    total_tasks: int
    evaluated_tasks: int

    @property
    def coverage_ratio(self) -> float:
        if self.total_tasks < 0 or self.evaluated_tasks < 0:
            raise ValueError("coverage counts must be non-negative")
        if self.evaluated_tasks > self.total_tasks:
            raise ValueError("evaluated_tasks cannot exceed total_tasks")
        if self.total_tasks == 0:
            return 0.0
        return self.evaluated_tasks / self.total_tasks
