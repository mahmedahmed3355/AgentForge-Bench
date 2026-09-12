from __future__ import annotations

import json

import pytest


gymnasium = pytest.importorskip("gymnasium")

from cuda_kernel_optimization_v1.gymnasium_env import (  # noqa: E402
    CudaKernelOptimizationGymEnv,
)


FORBIDDEN_AGENT_FACING_MARKERS = (
    "oracle_answer",
    "hidden_answer",
    "reference_solution",
    "reference_answer",
    "ground_truth",
    "hidden_solution",
    "secret_solution",
    "expected_optimal",
    "oracle_result",
    "verifier_result",
    "episode_report",
)


def _contains_forbidden(value: object) -> list[str]:
    blob = json.dumps(value, default=str).lower()

    return [
        marker
        for marker in FORBIDDEN_AGENT_FACING_MARKERS
        if marker in blob
    ]


def test_reset_info_does_not_expose_evaluator_report():
    env = CudaKernelOptimizationGymEnv(seed=1001)

    observation, info = env.reset()

    assert "task_id" in info
    assert "episode_report" not in info

    leaked = _contains_forbidden(
        {
            "observation": observation,
            "info": info,
        }
    )

    assert leaked == [], f"Agent-facing leakage detected: {leaked}"


def test_step_info_does_not_expose_evaluator_report():
    env = CudaKernelOptimizationGymEnv(seed=1001)

    env.reset()

    observation, reward, terminated, truncated, info = env.step(0)

    assert isinstance(info, dict)
    assert "action_name" in info
    assert "task_id" in info
    assert "episode_report" not in info

    leaked = _contains_forbidden(
        {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
    )

    assert leaked == [], f"Agent-facing leakage detected: {leaked}"


def test_full_episode_report_remains_evaluator_side():
    env = CudaKernelOptimizationGymEnv(seed=1001)

    env.reset()

    assert hasattr(env, "episode_report")

    # The evaluator-side report may exist on the environment object,
    # but it must never be copied into the agent-facing info payload.
    assert env.episode_report is not None

    _, _, _, _, info = env.step(0)

    assert "episode_report" not in info
