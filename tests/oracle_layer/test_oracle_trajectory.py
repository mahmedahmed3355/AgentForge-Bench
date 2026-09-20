import pytest

from agentforge.oracle import OracleStep, OracleTrajectory


def test_oracle_trajectory_preserves_order() -> None:
    first = OracleStep(action="a", observation="o1", reward=1.0)
    second = OracleStep(
        action="b",
        observation="o2",
        reward=2.5,
        terminated=True,
    )

    trajectory = OracleTrajectory(
        steps=(first, second)
    )

    assert len(trajectory) == 2
    assert trajectory.steps == (first, second)
    assert trajectory.total_reward == pytest.approx(3.5)
    assert trajectory.terminated is True


def test_append_returns_new_trajectory() -> None:
    first = OracleStep(action="a")
    second = OracleStep(action="b")

    original = OracleTrajectory((first,))
    updated = original.append(second)

    assert original.steps == (first,)
    assert updated.steps == (first, second)


@pytest.mark.parametrize(
    "reward",
    [float("nan"), float("inf"), float("-inf")],
)
def test_oracle_step_requires_finite_reward(
    reward: float,
) -> None:
    with pytest.raises(ValueError):
        OracleStep(action="bad", reward=reward)
