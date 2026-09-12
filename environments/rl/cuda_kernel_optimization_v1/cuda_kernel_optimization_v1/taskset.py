from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import verifiers.v1 as vf


@dataclass(frozen=True)
class KernelScenario:
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


class CudaKernelOptimizationTaskset(
    vf.Taskset[CudaKernelOptimizationTask, vf.TasksetConfig]
):
    def load(self) -> list[CudaKernelOptimizationTask]:
        scenarios = [
            KernelScenario(
                scenario_id="cuda-v1-train-001",
                seed=1001,
                matrix_size=256,
                baseline_ms=2.50,
                target_speedup=1.10,
            ),
            KernelScenario(
                scenario_id="cuda-v1-train-002",
                seed=1002,
                matrix_size=384,
                baseline_ms=4.20,
                target_speedup=1.15,
            ),
            KernelScenario(
                scenario_id="cuda-v1-eval-001",
                seed=2001,
                matrix_size=512,
                baseline_ms=7.80,
                target_speedup=1.20,
            ),
            KernelScenario(
                scenario_id="cuda-v1-hidden-001",
                seed=9001,
                matrix_size=768,
                baseline_ms=15.40,
                target_speedup=1.25,
            ),
        ]

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
                prompt=prompt,
                scenario_id=scenario.scenario_id,
                seed=scenario.seed,
                matrix_size=scenario.matrix_size,
                baseline_ms=scenario.baseline_ms,
                target_speedup=scenario.target_speedup,
            )

            tasks.append(
                CudaKernelOptimizationTask(task_data)
            )

        return tasks


__all__ = [
    "CudaKernelOptimizationData",
    "CudaKernelOptimizationTask",
    "CudaKernelOptimizationTaskset",
]
