
from agentforge.reporting import (
    TaskLevelResult,
    build_episode_summary,
    build_performance_from_rewards,
    build_task_coverage,
)


def test_build_episode_summary():
    summary = build_episode_summary(
        [
            {
                "reward": 2,
                "success": True,
                "length": 4,
            },
            {
                "reward": 0,
                "success": False,
                "length": 2,
                "truncated": True,
            },
        ]
    )

    assert summary.total_episodes == 2
    assert summary.successful_episodes == 1
    assert summary.truncated_episodes == 1
    assert summary.mean_reward == 1.0


def test_build_performance():
    metrics = build_performance_from_rewards(
        [1, 3],
        [True, False],
        [2, 4],
        evaluation_time_seconds=2,
    )

    assert metrics.success_rate == 0.5
    assert metrics.mean_reward == 2.0
    assert metrics.episodes_per_second == 1.0


def test_build_task_coverage():
    results = [
        TaskLevelResult(
            "a",
            "passed",
            1,
            1.0,
            1.0,
        ),
        TaskLevelResult(
            "b",
            "failed",
            1,
            0.0,
            0.0,
        ),
    ]

    coverage = build_task_coverage(
        results
    )

    assert coverage.total_tasks == 2
    assert coverage.passed_tasks == 1
    assert coverage.failed_tasks == 1
