import pytest

from agentforge.oracle import (
    OracleBehavior,
    OracleSolution,
    OracleStep,
    OracleTrajectory,
    build_solution,
)


def test_build_solution() -> None:
    trajectory = OracleTrajectory(
        steps=(
            OracleStep(action="a", reward=1.0),
            OracleStep(action="b", reward=2.0),
        )
    )

    behavior = OracleBehavior(
        name="demo",
        policy=lambda observation: observation,
    )

    solution = build_solution(
        trajectory,
        behavior,
    )

    assert isinstance(solution, OracleSolution)
    assert solution.actions() == ("a", "b")
    assert solution.total_reward == pytest.approx(3.0)


def test_solution_requires_valid_components() -> None:
    behavior = OracleBehavior(
        name="demo",
        policy=lambda observation: observation,
    )

    trajectory = OracleTrajectory()

    with pytest.raises(TypeError):
        build_solution("bad", behavior)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        build_solution(trajectory, "bad")  # type: ignore[arg-type]
