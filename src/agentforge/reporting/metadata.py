from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReportMetadata:
    benchmark_name: str
    version: str = "1.0"
    seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "version": self.version,
            "seed": self.seed,
            "extras": dict(self.extras),
        }
