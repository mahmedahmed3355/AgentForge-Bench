from __future__ import annotations

from verifiers.v1 import tool
from verifiers.v1.mcp import Toolset, ToolsetConfig

from .environment import CudaKernelOptimizationEnvironment
from .prime_state import CudaPrimeState


def _sync_state(
    state: CudaPrimeState,
    native_state,
    seed: int | None,
    observation: str,
) -> CudaPrimeState:
    """Copy public execution state from native CUDA state into Prime state."""

    state.seed = seed
    state.logical_stage = native_state.logical_stage
    state.action_count = native_state.action_count
    state.inspected = native_state.inspected
    state.candidate_modified = native_state.candidate_modified
    state.compilation_ok = native_state.compilation_ok
    state.correctness_ok = native_state.correctness_ok
    state.benchmark_completed = native_state.benchmark_completed
    state.analysis_completed = native_state.analysis_completed
    state.performance_improved = native_state.performance_improved
    state.regression_detected = native_state.regression_detected
    state.recovery_count = native_state.recovery_count
    state.terminal = native_state.terminal
    state.success = native_state.success
    state.finalized = native_state.finalized
    state.last_failure = native_state.last_failure
    state.last_observation = observation

    state.history.append(observation)

    return state


class CudaOptimizationToolset(Toolset[CudaPrimeState]):
    """Prime V1 tools for the CUDA kernel optimization environment."""

    TOOL_PREFIX = "cuda"

    def __init__(self, config: ToolsetConfig):
        super().__init__(config)
        self._env = CudaKernelOptimizationEnvironment()
        self._task_seed: int | None = None

    async def setup_task(self, task) -> None:
        """Initialize rollout-local task information from Prime V1."""
        self._task_seed = getattr(task.data, "seed", None)

    def _run_action(self, action: str) -> dict:
        state = self.state

        if state.action_count == 0:
            seed = state.seed if state.seed is not None else self._task_seed
            self._env.reset(seed=seed)
            state.seed = seed

        native_state, reward, done, info = self._env.step(action)

        observation = info.get(
            "observation",
            f"action={action}",
        )

        _sync_state(
            state=state,
            native_state=native_state,
            seed=state.seed,
            observation=str(observation),
        )

        safe_info = {
            str(key): (
                value.value
                if hasattr(value, "value")
                else value
            )
            for key, value in info.items()
        }

        return {
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "success": bool(native_state.success),
            "logical_stage": native_state.logical_stage,
            "action_count": native_state.action_count,
            "observation": str(observation),
            "info": safe_info,
        }

    @tool
    def inspect(self) -> dict:
        """Inspect the current CUDA optimization state and begin the episode."""
        return self._run_action("inspect")

    @tool
    def modify_memory(self) -> dict:
        """Apply a memory optimization candidate."""
        return self._run_action("modify_memory")

    @tool
    def modify_tiling(self) -> dict:
        """Apply a tiling optimization candidate."""
        return self._run_action("modify_tiling")

    @tool
    def modify_compute(self) -> dict:
        """Apply a compute optimization candidate."""
        return self._run_action("modify_compute")

    @tool
    def compile(self) -> dict:
        """Compile the current CUDA optimization candidate."""
        return self._run_action("compile")

    @tool
    def check_correctness(self) -> dict:
        """Check correctness of the current CUDA optimization candidate."""
        return self._run_action("check_correctness")

    @tool
    def benchmark(self) -> dict:
        """Benchmark the current CUDA optimization candidate."""
        return self._run_action("benchmark")

    @tool
    def analyze(self) -> dict:
        """Analyze the latest CUDA benchmark result."""
        return self._run_action("analyze")

    @tool
    def recover(self) -> dict:
        """Recover from a failed optimization path."""
        return self._run_action("recover")

    @tool
    def final_verify(self) -> dict:
        """Perform final verification of the CUDA optimization episode."""
        return self._run_action("final_verify")
