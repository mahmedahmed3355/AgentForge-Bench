import pytest

from agentforge.reward import RewardAggregator, RewardComponent


def test_aggregator_add_and_total() -> None:
    aggregator = RewardAggregator()

    aggregator.add(RewardComponent("progress", 2.0))
    aggregator.add(RewardComponent("quality", 3.0))

    assert aggregator.total() == pytest.approx(5.0)


def test_aggregator_extend() -> None:
    aggregator = RewardAggregator()

    aggregator.extend(
        [
            RewardComponent("a", 1.0),
            RewardComponent("b", 2.5),
        ]
    )

    assert len(aggregator.components) == 2
    assert aggregator.total() == pytest.approx(3.5)


def test_aggregator_rejects_wrong_type() -> None:
    aggregator = RewardAggregator()

    with pytest.raises(TypeError):
        aggregator.add("bad")  # type: ignore[arg-type]
