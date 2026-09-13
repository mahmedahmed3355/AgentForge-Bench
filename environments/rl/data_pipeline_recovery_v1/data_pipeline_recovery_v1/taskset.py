from __future__ import annotations

from dataclasses import dataclass

import verifiers.v1 as vf
from verifiers.v1.mcp import ToolsetConfig

from .decision_graph import IncidentKind
from .prime_toolset import PipelineRecoveryToolset


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str
    seed: int
    incident_family: str


class DataPipelineRecoveryData(vf.TaskData):
    scenario_id: str
    seed: int
    incident_family: str
    logical_stages: int = 18


class DataPipelineRecoveryTask(vf.Task[DataPipelineRecoveryData]):

    @vf.reward
    async def progress_reward(self, trace: vf.Trace) -> float:
        turns = trace.num_turns
        if turns <= 0:
            return 0.0
        return min(0.25, 0.05 * turns)

    @vf.stop
    async def horizon(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 40

    @classmethod
    def toolsets(cls, config) -> list:
        return [PipelineRecoveryToolset(ToolsetConfig())]


class DataPipelineRecoveryTaskset(
    vf.Taskset[
        DataPipelineRecoveryTask,
        vf.TasksetConfig,
    ]
):
    def load(self) -> list[DataPipelineRecoveryTask]:
        import json
        from importlib.resources import files

        scenario_file = files("data_pipeline_recovery_v1").joinpath(
            "data",
            "scenarios.json",
        )
        raw = json.loads(scenario_file.read_text())

        tasks = []

        for idx, scenario in enumerate(raw["scenarios"]):
            incident = IncidentKind(scenario["incident_family"])

            prompt = (
                "Recover the long-horizon data pipeline from the injected incident. "
                "Diagnose the causal failure, select the correct recovery branch and "
                "strategy, repair the affected pipeline, execute the pipeline, "
                "validate the output, recover or replan when required, and complete "
                "global verification. "
                f"Scenario={scenario['scenario_id']}; "
                f"incident={incident.value}; "
                f"seed={scenario['seed']}."
            )

            data = DataPipelineRecoveryData(
                idx=idx,
                name=f"Data Pipeline Recovery — {scenario['scenario_id']}",
                description=(
                    "Long-horizon stateful data pipeline recovery task with "
                    "causal dependencies, branching, recovery, replanning, "
                    "validation, and anti-shortcut final verification."
                ),
                prompt=prompt,
                scenario_id=scenario["scenario_id"],
                seed=int(scenario["seed"]),
                incident_family=incident.value,
                logical_stages=18,
            )

            tasks.append(DataPipelineRecoveryTask(data))

        return tasks
