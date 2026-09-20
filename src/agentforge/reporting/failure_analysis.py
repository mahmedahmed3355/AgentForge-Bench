from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FailureAnalysis:
    failures: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        if not reason:
            raise ValueError("failure reason must not be empty")
        self.failures.append(reason)

    @property
    def count(self) -> int:
        return len(self.failures)

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "failures": list(self.failures),
        }
