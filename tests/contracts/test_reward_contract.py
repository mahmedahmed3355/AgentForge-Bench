import math

import pytest

from agentforge.runtime.contracts.rewards import (
    RewardBreakdown,
    RewardComponent,
    RewardEngine,
)


def test_reward_component_accepts_finite_values():
    component = RewardComponent(
        name="correctness",
        value=2.0,
        weight=0.5,
    )

    assert component.name == "correctness"
    assert component.value == 2.0
    assert component.weight == 0.5


def test_reward_component_rejects_empty_name():
    with pytest.raises(ValueError):
        RewardComponent(
            name="",
            value=1.0,
        )

    with pytest.raises(ValueError):
        RewardComponent(
            name="   ",
            value=1.0,
        )


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_reward_component_rejects_non_finite_values(value):
    with pytest.raises(ValueError):
        RewardComponent(
            name="correctness",
            value=value,
        )

    with pytest.raises(ValueError):
        RewardComponent(
            name="correctness",
            value=1.0,
            weight=value,
        )


def test_reward_component_rejects_negative_weight():
    with pytest.raises(ValueError):
        RewardComponent(
            name="correctness",
            value=1.0,
            weight=-1.0,
        )


def test_reward_engine_calculates_multiple_components():
    engine = RewardEngine()

    breakdown = engine.calculate(
        (
            RewardComponent(
                name="correctness",
                value=2.0,
                weight=1.0,
            ),
            RewardComponent(
                name="efficiency",
                value=4.0,
                weight=0.25,
            ),
            RewardComponent(
                name="penalty",
                value=-2.0,
                weight=1.0,
            ),
        )
    )

    assert breakdown.total == pytest.approx(1.0)

    assert breakdown.weighted_components == {
        "correctness": pytest.approx(2.0),
        "efficiency": pytest.approx(1.0),
        "penalty": pytest.approx(-2.0),
    }


def test_reward_engine_rejects_duplicate_component_names():
    engine = RewardEngine()

    with pytest.raises(ValueError, match="duplicate"):
        engine.calculate(
            (
                RewardComponent(
                    name="correctness",
                    value=1.0,
                ),
                RewardComponent(
                    name="correctness",
                    value=2.0,
                ),
            )
        )


def test_reward_breakdown_rejects_inconsistent_total():
    components = (
        RewardComponent(
            name="correctness",
            value=2.0,
            weight=1.0,
        ),
        RewardComponent(
            name="efficiency",
            value=4.0,
            weight=0.5,
        ),
    )

    with pytest.raises(ValueError, match="does not equal"):
        RewardBreakdown(
            components=components,
            total=999.0,
        )


def test_reward_breakdown_rejects_duplicate_names():
    components = (
        RewardComponent(
            name="correctness",
            value=1.0,
        ),
        RewardComponent(
            name="correctness",
            value=2.0,
        ),
    )

    with pytest.raises(ValueError, match="unique"):
        RewardBreakdown.from_components(components)


def test_reward_engine_rejects_invalid_component_type():
    engine = RewardEngine()

    with pytest.raises(TypeError):
        engine.calculate(
            (
                "not-a-reward-component",
            )
        )


def test_reward_engine_supports_long_horizon_component_sets():
    engine = RewardEngine()

    components = tuple(
        RewardComponent(
            name=f"component_{index}",
            value=float(index + 1),
            weight=0.1,
        )
        for index in range(10)
    )

    breakdown = engine.calculate(components)

    expected = math.fsum(
        (index + 1) * 0.1
        for index in range(10)
    )

    assert breakdown.total == pytest.approx(expected)
    assert math.isfinite(breakdown.total)


def test_reward_engine_compatibility_aliases():
    engine = RewardEngine()

    components = (
        RewardComponent(
            name="progress",
            value=3.0,
            weight=2.0,
        ),
    )

    calculated = engine.calculate(components)
    computed = engine.compute(components)
    broken_down = engine.breakdown(components)

    assert calculated == computed
    assert calculated == broken_down
