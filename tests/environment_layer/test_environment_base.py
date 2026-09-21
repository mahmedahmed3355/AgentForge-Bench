from __future__ import annotations

import inspect

import gymnasium as gym
import pytest

from agentforge.contracts import EnvironmentContract
from agentforge.environment import AgentForgeEnv, EnvironmentState, Transition


class DummyEnv(AgentForgeEnv):
    """Minimal concrete environment for canonical lifecycle testing."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Discrete(3)
        self.action_space = gym.spaces.Discrete(2)

    def _reset_impl(
        self,
        *,
        seed: int | None,
        options: dict | None,
    ) -> tuple[int, dict]:
        return 0, {"reset": True}

    def _step_impl(self, action: int) -> Transition:
        return Transition(
            observation=1,
            reward=1.0,
            terminated=False,
            truncated=False,
            info={"action": action},
        )


def test_canonical_environment_inheritance() -> None:
    assert issubclass(AgentForgeEnv, gym.Env)
    assert issubclass(AgentForgeEnv, EnvironmentContract)


def test_environment_state_export() -> None:
    assert EnvironmentState is not None


def test_transition_export() -> None:
    assert Transition is not None


def test_reset_signature_is_canonical() -> None:
    signature = inspect.signature(AgentForgeEnv.reset)

    assert list(signature.parameters) == [
        "self",
        "seed",
        "options",
    ]

    assert signature.parameters["seed"].default is None
    assert signature.parameters["options"].default is None


def test_step_signature_is_canonical() -> None:
    signature = inspect.signature(AgentForgeEnv.step)

    assert list(signature.parameters) == [
        "self",
        "action",
    ]


def test_reset_lifecycle() -> None:
    env = DummyEnv()

    observation, info = env.reset(seed=42)

    assert observation in env.observation_space
    assert isinstance(info, dict)
    assert env.terminated is False
    assert env.truncated is False
    assert env.done is False


def test_step_lifecycle() -> None:
    env = DummyEnv()

    env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation in env.observation_space
    assert isinstance(reward, float)
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)


def test_no_step_before_reset() -> None:
    env = DummyEnv()

    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_invalid_action_rejected() -> None:
    env = DummyEnv()
    env.reset(seed=42)

    with pytest.raises((ValueError, AssertionError)):
        env.step(99)
