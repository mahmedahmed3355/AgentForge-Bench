import pytest

from agentforge.environment.base import NativeEnvironment
from agentforge.environment.transition import Transition


class DemoEnvironment(NativeEnvironment):
    def _reset_impl(self, *, seed, options):
        return (
            {"seed": seed, "options": options},
            {"phase": "initial"},
        )

    def _step_impl(self, action):
        if action == "finish":
            return Transition(
                observation={"status": "finished"},
                reward=1.0,
                terminated=True,
                truncated=False,
                info={"action": action},
            )

        if action == "timeout":
            return Transition(
                observation={"status": "timeout"},
                reward=0.0,
                terminated=False,
                truncated=True,
                info={"action": action},
            )

        return Transition(
            observation={"status": "running"},
            reward=0.1,
            terminated=False,
            truncated=False,
            info={"action": action},
        )


def test_reset_starts_episode() -> None:
    env = DemoEnvironment()

    observation, info = env.reset(seed=7)

    assert observation["seed"] == 7
    assert info["phase"] == "initial"
    assert env.state is not None
    assert env.state.step_count == 0
    assert env.done is False


def test_step_increments_step_count() -> None:
    env = DemoEnvironment()
    env.reset(seed=7)

    transition = env.step("continue")

    assert transition.reward == 0.1
    assert env.state is not None
    assert env.state.step_count == 1
    assert env.done is False


def test_terminal_transition_synchronizes_state() -> None:
    env = DemoEnvironment()
    env.reset(seed=7)

    transition = env.step("finish")

    assert transition.terminated is True
    assert env.terminated is True
    assert env.truncated is False
    assert env.done is True


def test_truncation_transition_synchronizes_state() -> None:
    env = DemoEnvironment()
    env.reset(seed=7)

    transition = env.step("timeout")

    assert transition.truncated is True
    assert env.terminated is False
    assert env.truncated is True
    assert env.done is True


def test_step_after_terminal_episode_is_rejected() -> None:
    env = DemoEnvironment()
    env.reset(seed=7)
    env.step("finish")

    with pytest.raises(RuntimeError):
        env.step("continue")


def test_step_after_truncation_is_rejected() -> None:
    env = DemoEnvironment()
    env.reset(seed=7)
    env.step("timeout")

    with pytest.raises(RuntimeError):
        env.step("continue")


def test_reset_starts_new_episode() -> None:
    env = DemoEnvironment()

    env.reset(seed=7)
    first_episode = env.episode_id
    env.step("finish")

    env.reset(seed=8)
    second_episode = env.episode_id

    assert first_episode != second_episode
    assert env.done is False
    assert env.state is not None
    assert env.state.step_count == 0


def test_close_clears_runtime_state() -> None:
    env = DemoEnvironment()
    env.reset(seed=7)

    assert env.state is not None

    env.close()

    assert env.state is None
    assert env.episode_id is None


def test_step_before_reset_is_rejected() -> None:
    env = DemoEnvironment()

    with pytest.raises(RuntimeError):
        env.step("continue")
