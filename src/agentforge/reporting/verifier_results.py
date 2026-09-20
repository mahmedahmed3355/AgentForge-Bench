from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerifierResults:
    total: int
    passed: int
    failed: int

    def __post_init__(self) -> None:
        if min(self.total, self.passed, self.failed) < 0:
            raise ValueError("verifier counts must be non-negative")
        if self.passed + self.failed != self.total:
            raise ValueError(
                "passed + failed must equal total"
            )

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total
