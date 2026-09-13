from __future__ import annotations

from dataclasses import dataclass

import verifiers.v1 as vf
from verifiers.v1.mcp import ToolsetConfig

from .prime_toolset import BranchingDecisionToolset
from .scenarios import load_public_scenarios


__all__ = [
    "BranchingDecisionRecoveryTaskset",
    "BranchingDecisionRecoveryTask",
    "BranchingDecisionRecoveryData",
]


@dataclass(frozen=True)
class _TasksetScenario:
    scenario_id: str
    seed: int
    scenario_family: str


class BranchingDecisionRecoveryData(vf.TaskData):
    scenario_id: str
    seed: int
    scenario_family: str
    logical_stages: int = 20


class BranchingDecisionRecoveryTask(
    vf.Task[BranchingDecisionRecoveryData]
):
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
    def toolsets(cls, config) -> list:
        return [
            BranchingDecisionToolset(
                ToolsetConfig()
            )
        ]


class BranchingDecisionRecoveryTaskset(
    vf.Taskset[
        BranchingDecisionRecoveryTask,
        vf.TasksetConfig,
    ]
):
    def load(self) -> list[BranchingDecisionRecoveryTask]:
        scenarios = load_public_scenarios()

        tasks: list[BranchingDecisionRecoveryTask] = []

        for idx, scenario in enumerate(scenarios):
            prompt = (
                "Solve the branching decision and recovery scenario. "
                "Gather useful information before committing to costly "
                "decisions. Form a hypothesis, select a branch and strategy, "
                "execute the plan, respond to delayed consequences, recover "
                "when necessary, replan, and perform global validation. "
                "Do not assume that local success implies final success. "
                f"Scenario={scenario.scenario_id}; "
                f"seed={scenario.seed}; "
                f"family={scenario.scenario_family}."
            )

            task_data = BranchingDecisionRecoveryData(
                idx=idx,
                name=(
                    "Branching Decision & Recovery — "
                    f"{scenario.scenario_id}"
                ),
                description=(
                    "Long-horizon decision-making task with partial "
                    "observability, costly branching, delayed consequences, "
                    "recovery, replanning, and global validation."
                ),
                prompt=prompt,
                scenario_id=scenario.scenario_id,
                seed=scenario.seed,
                scenario_family=scenario.scenario_family,
                logical_stages=20,
            )

            tasks.append(
                BranchingDecisionRecoveryTask(task_data)
            )

        return tasks
