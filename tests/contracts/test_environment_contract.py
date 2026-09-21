from __future__ import annotations

import math

import gymnasium as gym
import pytest

from agentforge.contracts import EnvironmentContract, StepResult


class DummyEnvironment(EnvironmentContract):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(2)
        self._next_observation = 1
        super().__init__()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        self._reset_episode_state()
        return 0, {}

    def _step_impl(self, action):
        return StepResult(
            observation=self._next_observation,
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"action": action},
        )


def test_step_result_has_canonical_fields():
    result = StepResult(
        observation=1,
        reward=1.0,
        terminated=False,
        truncated=False,
        info={},
    )

    assert result.observation == 1
    assert result.reward == 1.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info == {}


def test_terminated_and_truncated_cannot_both_be_true():
    with pytest.raises(ValueError):
        StepResult(
            observation=1,
            reward=1.0,
            terminated=True,
            truncated=True,
            info={},
        )


def test_reward_must_be_finite():
    with pytest.raises(ValueError):
        StepResult(
            observation=1,
            reward=math.inf,
            terminated=False,
            truncated=False,
            info={},
        )


def test_action_must_belong_to_action_space():
    env = DummyEnvironment()
    env.reset()

    with pytest.raises(ValueError):
        env.step(99)


def test_observation_must_belong_to_observation_space():
    class BadObservationEnvironment(DummyEnvironment):
        def _step_impl(self, action):
            return StepResult(
                observation=99,
                reward=1.0,
                terminated=True,
                truncated=False,
                info={},
            )

    env = BadObservationEnvironment()
    env.reset()

    with pytest.raises(ValueError):
        env.step(0)


def test_step_after_episode_end_requires_reset():
    env = DummyEnvironment()
    env.reset()

    env.step(0)

    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_reopens_episode():
    env = DummyEnvironment()

    env.reset()
    env.step(0)

    with pytest.raises(RuntimeError):
        env.step(0)

    observation, info = env.reset()

    assert observation == 0
    assert info == {}

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation == 1
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
