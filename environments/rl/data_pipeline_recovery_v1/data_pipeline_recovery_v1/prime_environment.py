from __future__ import annotations

import verifiers.v1 as vf


class DataPipelineRecoveryEnvConfig(vf.EnvConfig):
    """Prime V1 configuration for Data Pipeline Recovery V1."""

    agent: vf.AgentConfig = vf.AgentConfig()


class DataPipelineRecoveryPrimeEnv(
    vf.Env[DataPipelineRecoveryEnvConfig]
):
    """Prime V1 execution adapter for Data Pipeline Recovery V1."""

    async def run(self, task, agents) -> None:
        await agents.agent.run(task)
