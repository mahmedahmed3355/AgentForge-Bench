from __future__ import annotations

from .prime_toolset import CudaOptimizationToolset

from verifiers.v1.mcp import ToolsetConfig
from dataclasses import dataclass

import verifiers.v1 as vf

__all__ = ["CudaKernelOptimizationTaskset"]

from .scenarios import load_public_scenarios


@dataclass(frozen=True)
class _TasksetScenario:
    scenario_id: str
    seed: int
    matrix_size: int
    baseline_ms: float
    target_speedup: float


class CudaKernelOptimizationData(vf.TaskData):
    scenario_id: str
    seed: int
    matrix_size: int
    baseline_ms: float
    target_speedup: float
    logical_stages: int = 20


class CudaKernelOptimizationTask(vf.Task[CudaKernelOptimizationData]):
    @vf.reward
    async def progress_reward(self, trace: vf.Trace) -> float:
        turns = trace.num_turns
        if turns <= 0:
            return 0.0
        if trace.last_reply:
            return min(0.25, 0.05 * turns)
        return 0.0

    @vf.stop
    async def horizon(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 32



    @classmethod
    def toolsets(cls, config: ConfigT) -> list:
        """Return the task-scoped CUDA action tools for Prime V1."""
        return [CudaOptimizationToolset(ToolsetConfig())]

class CudaKernelOptimizationTaskset(
    vf.Taskset[CudaKernelOptimizationTask, vf.TasksetConfig]
):
    def load(self) -> list[CudaKernelOptimizationTask]:
        scenarios = load_public_scenarios()

        tasks: list[CudaKernelOptimizationTask] = []

        for idx, scenario in enumerate(scenarios):
            prompt = (
                "Optimize the CUDA kernel for the provided benchmark scenario. "
                "The episode contains 20 logical stages and may require recovery. "
                "The agent must preserve correctness while improving performance. "
                f"Scenario={scenario.scenario_id}; "
                f"matrix_size={scenario.matrix_size}; "
                f"baseline_ms={scenario.baseline_ms:.2f}; "
                f"target_speedup={scenario.target_speedup:.2f}."
            )

            task_data = CudaKernelOptimizationData(
                idx=idx,
                name=f"CUDA Kernel Optimization — {scenario.scenario_id}",
                description=(
                    "Long-horizon CUDA kernel optimization task with "
                    "branching, correctness constraints, performance targets, "
                    "and recovery."
                ),
                prompt=prompt,
                scenario_id=scenario.scenario_id,
                seed=scenario.seed,
                matrix_size=scenario.matrix_size,
                baseline_ms=scenario.baseline_ms,
                target_speedup=scenario.target_speedup,
            )

            tasks.append(CudaKernelOptimizationTask(task_data))

        return tasks
