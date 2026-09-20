import pytest

from agentforge.reward import RewardCalculation, RewardComponent


def test_calculation_from_components() -> None:
    components = (
        RewardComponent("progress", 2.0),
        RewardComponent("quality", 3.5),
    )

    calculation = RewardCalculation.from_components(components)

    assert calculation.total == pytest.approx(5.5)
    assert calculation.components == components


def test_calculation_rejects_non_finite_total() -> None:
    with pytest.raises(ValueError):
        RewardCalculation(float("inf"))
