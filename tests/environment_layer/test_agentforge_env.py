from __future__ import annotations

import inspect

import gymnasium as gym
import pytest
from gymnasium import spaces

from agentforge.contracts import EnvironmentContract
from agentforge.environment import AgentForgeEnv, EnvironmentState, Transition


class DummyEnv(AgentForgeEnv):
    observation_space = spaces.Discrete(3)
    action_space = spaces.Discrete(2)

    def _reset_impl(self, *, seed, options):
        return 0, {"reset": True}

    def _step_impl(self, action):
        if action == 1:
            return Transition(
                observation=1,
                reward=1.0,
                terminated=True,
                truncated=False,
                info={"done": True},
            )

        return Transition(
            observation=0,
            reward=0.5,
            terminated=False,
            truncated=False,
            info={"done": False},
        )


def test_canonical_inheritance():
    assert issubclass(AgentForgeEnv, gym.Env)
    assert issubclass(AgentForgeEnv, EnvironmentContract)


def test_required_task_hooks_exist():
    assert callable(getattr(AgentForgeEnv, "_reset_impl", None))
    assert callable(getattr(AgentForgeEnv, "_step_impl", None))


def test_reset_has_canonical_signature():
    env = DummyEnv()

    signature = inspect.signature(env.reset)

    assert signature.parameters["seed"].default is None
    assert signature.parameters["options"].default is None

    observation, info = env.reset(seed=7)

    assert observation == 0
    assert info == {"reset": True}
    assert env.state is not None
    assert isinstance(env.state, EnvironmentState)


def test_step_has_canonical_signature():
    env = DummyEnv()
    env.reset()

    result = env.step(0)

    assert isinstance(result, tuple)
    assert len(result) == 5

    observation, reward, terminated, truncated, info = result

    assert observation == 0
    assert reward == 0.5
    assert terminated is False
    assert truncated is False
    assert info == {"done": False}


def test_action_space_invariant():
    env = DummyEnv()
    env.reset()

    with pytest.raises(ValueError, match="action_space"):
        env.step(99)


def test_no_step_after_episode_end():
    env = DummyEnv()
    env.reset()

    observation, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False

    with pytest.raises(
        RuntimeError,
        match="call reset\\(\\) first",
    ):
        env.step(0)


def test_observation_space_invariant():
    class InvalidObservationEnv(DummyEnv):
        def _reset_impl(self, *, seed, options):
            return 99, {}

    env = InvalidObservationEnv()

    with pytest.raises(
        ValueError,
        match="observation_space",
    ):
        env.reset()


def test_terminated_truncated_invariant():
    class InvalidTransitionEnv(DummyEnv):
        def _step_impl(self, action):
            return Transition(
                observation=0,
                reward=1.0,
                terminated=True,
                truncated=True,
                info={},
            )

    env = InvalidTransitionEnv()
    env.reset()

    with pytest.raises(
        ValueError,
        match="terminated and truncated",
    ):
        env.step(0)


def test_finite_reward_invariant():
    class InvalidRewardEnv(DummyEnv):
        def _step_impl(self, action):
            return Transition(
                observation=0,
                reward=float("nan"),
                terminated=False,
                truncated=False,
                info={},
            )

    env = InvalidRewardEnv()
    env.reset()

    with pytest.raises(
        ValueError,
        match="finite scalar",
    ):
        env.step(0)
