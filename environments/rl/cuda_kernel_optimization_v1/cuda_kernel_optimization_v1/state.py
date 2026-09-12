from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KernelOptimizationState:
    logical_stage: int = 0
    step_count: int = 0
    candidate_modified: bool = False
    compilation_ok: bool = False
    correctness_ok: bool = False
    benchmark_completed: bool = False
    performance_improved: bool = False
    analysis_completed: bool = False
    recovery_count: int = 0
    finalized: bool = False
    failure_history: list[str] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.finalized

    def advance(self) -> None:
        self.logical_stage = min(20, self.logical_stage + 1)
        self.step_count += 1

    def fail(self, reason: str) -> None:
        self.failure_history.append(reason)
        self.step_count += 1

    def recover(self) -> None:
        self.recovery_count += 1
        self.step_count += 1
