import math

import pytest

from agentforge.reward import RewardComponent


def test_reward_component_is_constructible() -> None:
    component = RewardComponent("progress", 1.5)

    assert component.name == "progress"
    assert component.value == 1.5


def test_reward_component_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        RewardComponent("", 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_reward_component_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        RewardComponent("invalid", value)
