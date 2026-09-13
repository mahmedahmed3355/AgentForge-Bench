from __future__ import annotations

import verifiers.v1 as vf


class BranchingDecisionRecoveryEnvConfig(
    vf.EnvConfig
):
    """Prime V1 configuration for Task 003."""

    agent: vf.AgentConfig = vf.AgentConfig()


class BranchingDecisionRecoveryPrimeEnv(
    vf.Env[BranchingDecisionRecoveryEnvConfig]
):
    """Prime V1 execution adapter for Task 003."""

    async def run(self, task, agents) -> None:
        await agents.agent.run(task)
