import pytest

from agentforge.environment.state import EnvironmentState


def test_initial_state_is_not_done() -> None:
    state = EnvironmentState("episode-1")
    assert state.step_count == 0
    assert state.terminated is False
    assert state.truncated is False
    assert state.done is False


def test_terminated_state_is_done() -> None:
    state = EnvironmentState("episode-1")
    state.mark_terminated()

    assert state.terminated is True
    assert state.truncated is False
    assert state.done is True


def test_truncated_state_is_done() -> None:
    state = EnvironmentState("episode-1")
    state.mark_truncated()

    assert state.truncated is True
    assert state.terminated is False
    assert state.done is True


def test_both_terminal_flags_are_rejected() -> None:
    with pytest.raises(ValueError):
        EnvironmentState(
            "episode-1",
            terminated=True,
            truncated=True,
        )
