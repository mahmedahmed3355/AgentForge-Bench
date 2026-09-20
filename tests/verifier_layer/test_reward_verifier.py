import math

from agentforge.verifier.reward import verify_reward


def test_finite_reward_is_valid():
    result = verify_reward(1.5)
    assert result.valid is True
    assert result.value == 1.5


def test_nan_reward_is_rejected():
    result = verify_reward(math.nan)
    assert result.valid is False


def test_infinite_reward_is_rejected():
    result = verify_reward(math.inf)
    assert result.valid is False


def test_numeric_string_is_accepted():
    result = verify_reward("2.5")
    assert result.valid is True
    assert result.value == 2.5
