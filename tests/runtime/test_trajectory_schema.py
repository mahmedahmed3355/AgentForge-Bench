from agentforge.runtime.contracts.rewards import (
    RewardBreakdown,
    RewardComponent,
)
from agentforge.runtime.trajectory import (
    Trajectory,
    TrajectoryStep,
)


def reward_breakdown():
    return RewardBreakdown(
        components=(
            RewardComponent(
                name="correctness",
                value=1.0,
                weight=1.0,
            ),
        ),
        total=1.0,
    )


def make_step(index=0):
    return TrajectoryStep(
        episode_id="episode-001",
        task_id="task-001",
        scenario_id="scenario-001",
        seed=42,
        step_index=index,
        observation={"state": index},
        action={"action": "continue"},
        reward_total=1.0,
        reward_components=reward_breakdown(),
        terminated=False,
        truncated=False,
        info={"stage": "execution"},
    )


def test_trajectory_step_has_canonical_schema():
    step = make_step()

    assert step.episode_id == "episode-001"
    assert step.task_id == "task-001"
    assert step.scenario_id == "scenario-001"
    assert step.seed == 42
    assert step.step_index == 0
    assert step.observation == {"state": 0}
    assert step.action == {"action": "continue"}
    assert step.reward_total == 1.0
    assert step.reward_components.total == 1.0
    assert step.terminated is False
    assert step.truncated is False
    assert step.info["stage"] == "execution"


def test_trajectory_rejects_mixed_episode_identity():
    first = make_step(0)
    second = TrajectoryStep(
        episode_id="different-episode",
        task_id=first.task_id,
        scenario_id=first.scenario_id,
        seed=first.seed,
        step_index=1,
        observation={},
        action={},
        reward_total=0.0,
        reward_components=reward_breakdown(),
        terminated=False,
        truncated=False,
        info={},
    )

    try:
        Trajectory(
            episode_id=first.episode_id,
            task_id=first.task_id,
            scenario_id=first.scenario_id,
            seed=first.seed,
            steps=(first, second),
        )
    except ValueError as exc:
        assert "episode_id" in str(exc)
    else:
        raise AssertionError(
            "Trajectory accepted mixed episode identities."
        )


def test_trajectory_requires_strict_step_order():
    first = make_step(1)
    second = make_step(1)

    try:
        Trajectory(
            episode_id="episode-001",
            task_id="task-001",
            scenario_id="scenario-001",
            seed=42,
            steps=(first, second),
        )
    except ValueError as exc:
        assert "step_index" in str(exc)
    else:
        raise AssertionError(
            "Trajectory accepted duplicate step indexes."
        )


def test_trajectory_append_preserves_identity():
    trajectory = Trajectory(
        episode_id="episode-001",
        task_id="task-001",
        scenario_id="scenario-001",
        seed=42,
    )

    trajectory = trajectory.append(make_step(0))
    trajectory = trajectory.append(make_step(1))

    assert trajectory.length == 2
    assert trajectory.steps[0].step_index == 0
    assert trajectory.steps[1].step_index == 1
    assert trajectory.total_reward == 2.0


def test_terminated_and_truncated_are_distinct():
    terminated = TrajectoryStep(
        episode_id="episode-001",
        task_id="task-001",
        scenario_id="scenario-001",
        seed=42,
        step_index=0,
        observation={},
        action={},
        reward_total=0.0,
        reward_components=reward_breakdown(),
        terminated=True,
        truncated=False,
        info={},
    )

    truncated = TrajectoryStep(
        episode_id="episode-002",
        task_id="task-001",
        scenario_id="scenario-001",
        seed=42,
        step_index=0,
        observation={},
        action={},
        reward_total=0.0,
        reward_components=reward_breakdown(),
        terminated=False,
        truncated=True,
        info={},
    )

    assert terminated.terminated is True
    assert terminated.truncated is False
    assert truncated.terminated is False
    assert truncated.truncated is True


def test_terminated_and_truncated_cannot_both_be_true():
    try:
        TrajectoryStep(
            episode_id="episode-001",
            task_id="task-001",
            scenario_id="scenario-001",
            seed=42,
            step_index=0,
            observation={},
            action={},
            reward_total=0.0,
            reward_components=reward_breakdown(),
            terminated=True,
            truncated=True,
            info={},
        )
    except ValueError as exc:
        assert "terminated" in str(exc)
    else:
        raise AssertionError(
            "TrajectoryStep accepted both termination flags as true."
        )
