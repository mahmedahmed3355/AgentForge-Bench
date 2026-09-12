from __future__ import annotations

from copy import deepcopy

from dataclasses import dataclass, field
from enum import Enum

from .decision_graph import DecisionKind, get_branch_outcome
from .trajectory import Trajectory, Transition


class FailureType(str, Enum):
    NONE = "none"
    COMPILATION = "compilation"
    CORRECTNESS = "correctness"
    REGRESSION = "regression"


# Backward-compatible public name for existing Task 001 tests/API.
OptimizationBranch = DecisionKind



@dataclass
class CudaEnvironmentState:

    logical_stage: int = 0
    action_count: int = 0
    inspected: bool = False
    optimization_branch: DecisionKind | None = None
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
    finalized: bool = False
    history: list[str] = field(default_factory=list)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


class CudaKernelOptimizationEnvironment:
    MAX_ACTIONS = 32
    LOGICAL_STAGES = 20

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.state = CudaEnvironmentState()
        self.trajectory = Trajectory()

    def reset(self, seed: int | None = None) -> CudaEnvironmentState:
        if seed is not None:
            self.seed = seed

        self.state = CudaEnvironmentState()
        self.trajectory = Trajectory()
        return deepcopy(self.state)

    def _record(
        self,
        action: str,
        reward: float,
        observation: str,
        stage_before: int | None = None,
    ) -> None:
        self.trajectory.append(
            step=self.state.action_count,
            stage_before=(
                self.state.logical_stage
                if stage_before is None
                else stage_before
            ),
            stage_after=self.state.logical_stage,
            action=action,
            reward=reward,
            success=self.state.success,
            terminal=self.state.terminal,
            observation=observation,
            info={
                "decision": (
                    self.state.optimization_branch.value
                    if self.state.optimization_branch is not None
                    else None
                ),
                "failure": self.state.last_failure,
                "recovery_count": self.state.recovery_count,
            },
        )

    def _advance(self, amount: int) -> None:
        self.state.logical_stage = min(
            self.LOGICAL_STAGES,
            self.state.logical_stage + amount,
        )

    def step(self, action: str):
        stage_before = self.state.logical_stage

        if self.state.terminal:
            return self.state, 0.0, True, {
                "reason": "terminal",
            }

        if self.state.action_count >= self.MAX_ACTIONS:
            self.state.terminal = True
            self.state.success = False
            self._record(action, -1.0, "maximum action budget reached")
            return self.state, -1.0, True, {
                "reason": "max_actions",
            }

        self.state.action_count += 1
        reward = 0.0
        observation = ""

        if action == "inspect":
            self.state.inspected = True
            self._advance(2)
            reward = 1.0
            observation = "kernel inspected"

        elif action == "modify_memory":
            if not self.state.inspected:
                reward = -1.0
                observation = "invalid: inspect required"
            else:
                self.state.optimization_branch = DecisionKind.MEMORY
                self.state.candidate_modified = True
                self._advance(3)
                reward = 1.5
                observation = "memory optimization candidate applied"

        elif action == "modify_tiling":
            if not self.state.inspected:
                reward = -1.0
                observation = "invalid: inspect required"
            else:
                self.state.optimization_branch = DecisionKind.TILING
                self.state.candidate_modified = True
                self._advance(3)
                reward = 1.5
                observation = "tiling optimization candidate applied"

        elif action == "modify_compute":
            if not self.state.inspected:
                reward = -1.0
                observation = "invalid: inspect required"
            else:
                self.state.optimization_branch = DecisionKind.COMPUTE
                self.state.candidate_modified = True
                self._advance(3)
                reward = 1.5
                observation = "compute optimization candidate applied"

        elif action == "compile":
            if not self.state.candidate_modified:
                reward = -1.0
                observation = "invalid: candidate modification required"
            else:
                branch = self.state.optimization_branch
                outcome = get_branch_outcome(branch)

                if outcome.compile_success:
                    self.state.compilation_ok = True
                    self._advance(3)
                    reward = 2.0
                    observation = "compilation succeeded"
                else:
                    self.state.compilation_ok = False
                    self.state.last_failure = FailureType.COMPILATION
                    self.state.regression_detected = outcome.delayed_regression
                    self._advance(1)
                    reward = -2.0
                    observation = "compilation failed; recovery required"

        elif action == "check_correctness":
            if not self.state.compilation_ok:
                reward = -1.5
                observation = "invalid: successful compilation required"
            else:
                branch = self.state.optimization_branch
                outcome = get_branch_outcome(branch)

                if outcome.correctness_success:
                    self.state.correctness_ok = True
                    self._advance(3)
                    reward = 2.5
                    observation = "correctness checks passed"
                else:
                    self.state.correctness_ok = False
                    self.state.last_failure = FailureType.CORRECTNESS
                    self._advance(1)
                    reward = -2.0
                    observation = "correctness failure detected"

        elif action == "benchmark":
            if not self.state.correctness_ok:
                reward = -1.5
                observation = "invalid: correctness must pass first"
            else:
                branch = self.state.optimization_branch
                outcome = get_branch_outcome(branch)

                if outcome.benchmark_success:
                    self.state.benchmark_completed = True
                    self.state.performance_improved = True
                    self._advance(2)
                    reward = 3.0
                    observation = "benchmark completed with improvement"
                else:
                    self.state.benchmark_completed = False
                    self._advance(1)
                    reward = -1.0
                    observation = "benchmark did not establish improvement"

        elif action == "analyze":
            if not self.state.benchmark_completed:
                reward = -1.0
                observation = "invalid: benchmark required"
            else:
                self.state.analysis_completed = True
                self._advance(2)
                reward = 1.5
                observation = "benchmark results analyzed"

        elif action == "recover":
            if self.state.last_failure == FailureType.NONE:
                reward = -1.0
                observation = "invalid: no recoverable failure"
            else:
                self.state.recovery_count += 1
                self.state.last_failure = FailureType.NONE
                self.state.candidate_modified = False
                self.state.compilation_ok = False
                self.state.correctness_ok = False
                self.state.benchmark_completed = False
                self.state.analysis_completed = False
                self.state.performance_improved = False
                self.state.regression_detected = False
                self._advance(1)
                reward = 1.0
                observation = "recovery completed; replanning allowed"

        elif action == "final_verify":
            if (
                self.state.logical_stage >= self.LOGICAL_STAGES
                and self.state.correctness_ok
                and self.state.performance_improved
                and self.state.analysis_completed
            ):
                self.state.success = True
                self.state.finalized = True
                self.state.terminal = True
                reward = 10.0
                observation = "final verification passed"
            else:
                reward = -2.0
                observation = "final verification failed"

        else:
            reward = -1.0
            observation = f"invalid action: {action}"

        self.state.history.append(action)
        self._record(action, reward, observation, stage_before)

        done = self.state.terminal
        return self.state, reward, done, {
            "failure": self.state.last_failure,
            "performance_improved": self.state.performance_improved,
            "regression_detected": self.state.regression_detected,
            "decision": (
                self.state.optimization_branch.value
                if self.state.optimization_branch is not None
                else None
            ),
            "observation": observation,
            "logical_stage": self.state.logical_stage,
            "action_count": self.state.action_count,
            "recovery_count": self.state.recovery_count,
            "success": self.state.success,
        }
