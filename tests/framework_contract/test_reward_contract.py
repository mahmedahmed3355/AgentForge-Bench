import math

import pytest

from agentforge.contracts.reward import RewardContract


def test_reward_accepts_finite_scalar() -> None:
    contract = RewardContract()

    assert contract.validate(1.25) == 1.25
    assert contract.validate(-2) == -2.0
    assert contract.validate(0) == 0.0


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_reward_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError):
        RewardContract().validate(value)


def test_reward_rejects_boolean() -> None:
    with pytest.raises(ValueError):
        RewardContract().validate(True)


def test_non_negative_reward_mode() -> None:
    contract = RewardContract(allow_negative=False)

    with pytest.raises(ValueError):
        contract.validate(-1.0)
