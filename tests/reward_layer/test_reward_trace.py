import pytest

from agentforge.reward import RewardComponent, RewardTrace


def test_trace_preserves_component_order() -> None:
    trace = RewardTrace()

    first = RewardComponent("first", 1.0)
    second = RewardComponent("second", 2.0)

    trace.add(first)
    trace.add(second)

    assert trace.as_tuple() == (first, second)
    assert trace.total == pytest.approx(3.0)


def test_trace_extend() -> None:
    trace = RewardTrace()

    trace.extend(
        [
            RewardComponent("a", 1.25),
            RewardComponent("b", 2.75),
        ]
    )

    assert trace.total == pytest.approx(4.0)
