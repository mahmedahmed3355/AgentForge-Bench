from __future__ import annotations

import verifiers.v1 as vf


class CudaKernelOptimizationEnvConfig(vf.EnvConfig):
    """Prime V1 configuration for CUDA Kernel Optimization V1."""

    agent: vf.AgentConfig = vf.AgentConfig()


class CudaKernelOptimizationEnv(vf.Env[CudaKernelOptimizationEnvConfig]):
    """Prime V1 execution adapter for CUDA Kernel Optimization V1."""

    async def run(self, task, agents) -> None:
        await agents.agent.run(task)
