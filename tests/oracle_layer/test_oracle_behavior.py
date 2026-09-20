from agentforge.oracle import (
    OracleBehavior,
    OracleStep,
    OracleTrajectory,
    behavior_matches_trajectory,
    replay_behavior,
)


def test_oracle_behavior_calls_reference_policy() -> None:
    behavior = OracleBehavior(
        name="demo",
        policy=lambda observation: observation["action"],
    )

    observations = [
        {"action": "a"},
        {"action": "b"},
    ]

    assert replay_behavior(behavior, observations) == [
        "a",
        "b",
    ]


def test_behavior_matches_reference_trajectory() -> None:
    behavior = OracleBehavior(
        name="demo",
        policy=lambda observation: observation,
    )

    trajectory = OracleTrajectory(
        steps=(
            OracleStep(action="a"),
            OracleStep(action="b"),
        )
    )

    assert behavior_matches_trajectory(
        behavior,
        ["a", "b"],
        trajectory,
    )


def test_behavior_mismatch_is_false() -> None:
    behavior = OracleBehavior(
        name="demo",
        policy=lambda observation: observation,
    )

    trajectory = OracleTrajectory(
        steps=(
            OracleStep(action="a"),
            OracleStep(action="b"),
        )
    )

    assert not behavior_matches_trajectory(
        behavior,
        ["a", "wrong"],
        trajectory,
    )
