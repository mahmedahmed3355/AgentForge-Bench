import pytest


gym = pytest.importorskip("gymnasium")

from environments.rl.cuda_kernel_optimization_v1.cuda_kernel_optimization_v1.gymnasium_env import (
    ACTION_NAMES,
    CudaKernelOptimizationGymEnv,
)


def test_gymnasium_spaces_exist():
    env = CudaKernelOptimizationGymEnv(seed=123)

    assert env.action_space.n == len(ACTION_NAMES)
    assert env.observation_space.contains(
        env.reset()[0]
    )

    env.close()


def test_gymnasium_step_contract():
    env = CudaKernelOptimizationGymEnv(seed=123)

    observation, info = env.reset()

    assert isinstance(observation, dict)
    assert "logical_stage" in observation
    assert info["task_id"] == "cuda-kernel-optimization-v1"

    observation, reward, terminated, truncated, info = env.step(0)

    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["action_name"] == ACTION_NAMES[0]

    env.close()


def test_gymnasium_report_tracks_actions():
    env = CudaKernelOptimizationGymEnv(seed=123)

    env.reset()

    env.step(0)
    env.step(2)

    report = env.episode_report

    assert report.action_count == 2
    assert len(report.trajectory) == 2
    assert report.trajectory[0].action == ACTION_NAMES[0]
    assert report.trajectory[1].action == ACTION_NAMES[2]

    env.close()
