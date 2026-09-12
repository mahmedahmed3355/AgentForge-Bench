from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class OptimizationBranch(StrEnum):
    MEMORY = "memory"
    TILING = "tiling"
    COMPUTE = "compute"


class FailureType(StrEnum):
    NONE = "none"
    COMPILATION = "compilation"
    CORRECTNESS = "correctness"
    REGRESSION = "regression"


@dataclass
class CudaEnvironmentState:
    logical_stage: int = 0
    action_count: int = 0

    inspected: bool = False
    optimization_branch: OptimizationBranch | None = None

    candidate_modified: bool = False
    compilation_ok: bool = False
    correctness_ok: bool = False
    benchmark_completed: bool = False
    analysis_completed: bool = False

    performance_improved: bool = False
    regression_detected: bool = False

    last_failure: FailureType = FailureType.NONE
    recovery_count: int = 0

    terminal: bool = False
    success: bool = False

    history: list[str] = field(default_factory=list)

    def record(self, event: str) -> None:
        self.history.append(event)

    def advance_stage(self, amount: int = 1) -> None:
        self.logical_stage = min(20, self.logical_stage + amount)

    def next_action(self) -> None:
        self.action_count += 1

    def fail(self, failure: FailureType, reason: str) -> None:
        self.last_failure = failure
        self.record(reason)

    def recover(self, reason: str) -> None:
        self.recovery_count += 1
        self.last_failure = FailureType.NONE
        self.regression_detected = False
        self.record(reason)


class CudaKernelOptimizationEnvironment:
    MAX_ACTIONS = 32
    LOGICAL_STAGES = 20

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.state = CudaEnvironmentState()

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = seed

        self.state = CudaEnvironmentState()
        self.state.record(f"reset:seed={self.seed}")
        return self.observe()

    def observe(self) -> dict:
        available_actions = [
            "inspect",
            "modify_memory",
            "modify_tiling",
            "modify_compute",
            "compile",
            "check_correctness",
            "benchmark",
            "analyze",
            "recover",
            "final_verify",
        ]

        if self.state.terminal:
            available_actions = []

        return {
            "logical_stage": self.state.logical_stage,
            "action_count": self.state.action_count,
            "inspected": self.state.inspected,
            "optimization_branch": (
                self.state.optimization_branch.value
                if self.state.optimization_branch
                else None
            ),
            "candidate_modified": self.state.candidate_modified,
            "compilation_ok": self.state.compilation_ok,
            "correctness_ok": self.state.correctness_ok,
            "benchmark_completed": self.state.benchmark_completed,
            "analysis_completed": self.state.analysis_completed,
            "performance_improved": self.state.performance_improved,
            "regression_detected": self.state.regression_detected,
            "last_failure": self.state.last_failure.value,
            "recovery_count": self.state.recovery_count,
            "terminal": self.state.terminal,
            "success": self.state.success,
            "available_actions": available_actions,
        }

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if self.state.terminal:
            return self.observe(), -1.0, True, {
                "error": "episode_already_terminal"
            }

        self.state.next_action()

        reward = 0.0
        info: dict = {}

        if action == "inspect":
            reward = self._inspect()

        elif action == "modify_memory":
            reward = self._modify(OptimizationBranch.MEMORY)

        elif action == "modify_tiling":
            reward = self._modify(OptimizationBranch.TILING)

        elif action == "modify_compute":
            reward = self._modify(OptimizationBranch.COMPUTE)

        elif action == "compile":
            reward = self._compile()

        elif action == "check_correctness":
            reward = self._check_correctness()

        elif action == "benchmark":
            reward = self._benchmark()

        elif action == "analyze":
            reward = self._analyze()

        elif action == "recover":
            reward = self._recover()

        elif action == "final_verify":
            reward = self._final_verify()

        else:
            reward = -0.50
            info["invalid_action"] = action
            self.state.record(f"invalid:{action}")

        if self.state.action_count >= self.MAX_ACTIONS and not self.state.terminal:
            self.state.terminal = True
            self.state.success = False
            info["truncated"] = True

        return (
            self.observe(),
            reward,
            self.state.terminal,
            info,
        )

    def _inspect(self) -> float:
        if self.state.inspected:
            self.state.record("inspect:repeat")
            return -0.05

        self.state.inspected = True
        self.state.advance_stage(2)
        self.state.record("inspect:complete")
        return 0.20

    def _modify(self, branch: OptimizationBranch) -> float:
        if not self.state.inspected:
            self.state.record("modify:before_inspect")
            return -0.20

        if self.state.candidate_modified:
            self.state.record("modify:repeat")
            return -0.05

        self.state.optimization_branch = branch
        self.state.candidate_modified = True
        self.state.compilation_ok = False
        self.state.correctness_ok = False
        self.state.benchmark_completed = False
        self.state.analysis_completed = False
        self.state.performance_improved = False
        self.state.regression_detected = False

        self.state.advance_stage(3)
        self.state.record(f"modify:{branch.value}")
        return 0.50

    def _compile(self) -> float:
        if not self.state.candidate_modified:
            self.state.fail(
                FailureType.COMPILATION,
                "compile:missing_candidate",
            )
            return -0.30

        if self.state.optimization_branch == OptimizationBranch.COMPUTE:
            self.state.fail(
                FailureType.COMPILATION,
                "compile:compute_branch_failure",
            )
            self.state.compilation_ok = False
            return -0.25

        self.state.compilation_ok = True
        self.state.advance_stage(3)
        self.state.record("compile:success")
        return 0.50

    def _check_correctness(self) -> float:
        if not self.state.compilation_ok:
            self.state.fail(
                FailureType.CORRECTNESS,
                "correctness:compile_required",
            )
            return -0.25

        if self.state.optimization_branch == OptimizationBranch.MEMORY:
            self.state.fail(
                FailureType.CORRECTNESS,
                "correctness:memory_branch_failure",
            )
            self.state.correctness_ok = False
            return -0.40

        self.state.correctness_ok = True
        self.state.advance_stage(3)
        self.state.record("correctness:pass")
        return 0.75

    def _benchmark(self) -> float:
        if not self.state.correctness_ok:
            self.state.record("benchmark:blocked")
            return -0.25

        self.state.benchmark_completed = True
        self.state.advance_stage(2)

        if self.state.optimization_branch == OptimizationBranch.TILING:
            self.state.performance_improved = True
            self.state.regression_detected = False
            self.state.record("benchmark:improvement")
            return 0.75

        self.state.performance_improved = False
        self.state.regression_detected = True
        self.state.fail(
            FailureType.REGRESSION,
            "benchmark:performance_regression",
        )
        return -0.35

    def _analyze(self) -> float:
        if not self.state.benchmark_completed:
            self.state.record("analyze:benchmark_required")
            return -0.20

        self.state.analysis_completed = True
        self.state.advance_stage(2)
        self.state.record("analyze:complete")

        if self.state.regression_detected:
            return 0.10

        return 0.25

    def _recover(self) -> float:
        if self.state.last_failure == FailureType.NONE and not self.state.regression_detected:
            self.state.record("recover:nothing_to_recover")
            return -0.10

        self.state.recover("recover:complete")

        self.state.candidate_modified = False
        self.state.compilation_ok = False
        self.state.correctness_ok = False
        self.state.benchmark_completed = False
        self.state.analysis_completed = False
        self.state.performance_improved = False

        self.state.advance_stage(1)
        return 0.30

    def _final_verify(self) -> float:
        if (
            self.state.logical_stage >= self.LOGICAL_STAGES
            and self.state.correctness_ok
            and self.state.performance_improved
            and self.state.analysis_completed
        ):
            self.state.logical_stage = self.LOGICAL_STAGES
            self.state.terminal = True
            self.state.success = True
            self.state.record("final_verify:success")
            return 5.0

        self.state.record("final_verify:failure")
        return -0.50


__all__ = [
    "CudaEnvironmentState",
    "CudaKernelOptimizationEnvironment",
    "FailureType",
    "OptimizationBranch",
]
