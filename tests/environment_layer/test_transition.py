import pytest

from agentforge.environment.transition import Transition


def test_valid_transition() -> None:
    transition = Transition(
        observation={"value": 1},
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"step": 1},
    )

    assert transition.reward == 1.0
    assert transition.terminated is False
    assert transition.truncated is False


def test_non_finite_reward_is_rejected() -> None:
    with pytest.raises(ValueError):
        Transition(
            observation={},
            reward=float("nan"),
            terminated=False,
            truncated=False,
            info={},
        )


def test_both_terminal_flags_are_rejected() -> None:
    with pytest.raises(ValueError):
        Transition(
            observation={},
            reward=0.0,
            terminated=True,
            truncated=True,
            info={},
        )


def test_info_must_be_dict() -> None:
    with pytest.raises(TypeError):
        Transition(
            observation={},
            reward=0.0,
            terminated=False,
            truncated=False,
            info=None,
        )
