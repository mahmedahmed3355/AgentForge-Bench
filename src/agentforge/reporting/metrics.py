
from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import sqrt
from typing import Any

from .benchmark_report import (
    EpisodeSummary,
    PerformanceMetrics,
    TaskCoverage,
    TaskLevelResult,
)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0

    mean = _mean(values)

    return sqrt(
        sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)
    )


def build_task_coverage(
    results: Iterable[TaskLevelResult],
    total_tasks: int | None = None,
) -> TaskCoverage:

    items = list(results)

    total = (
        len(items)
        if total_tasks is None
        else total_tasks
    )

    evaluated = sum(
        item.episodes > 0
        for item in items
    )

    passed = sum(
        item.status.lower()
        in {
            "pass",
            "passed",
            "success",
            "successful",
        }
        for item in items
    )

    failed = sum(
        item.status.lower()
        in {
            "fail",
            "failed",
            "error",
            "unsuccessful",
        }
        for item in items
    )

    skipped = max(
        total - evaluated,
        0,
    )

    return TaskCoverage(
        total_tasks=total,
        evaluated_tasks=evaluated,
        passed_tasks=passed,
        failed_tasks=failed,
        skipped_tasks=skipped,
    )


def build_episode_summary(
    episodes: Iterable[Mapping[str, Any]],
) -> EpisodeSummary:

    items = list(episodes)

    rewards = [
        float(item.get("reward", 0.0))
        for item in items
    ]

    lengths = [
        float(
            item.get(
                "length",
                item.get(
                    "episode_length",
                    0.0,
                ),
            )
        )
        for item in items
    ]

    successful = sum(
        bool(
            item.get(
                "success",
                item.get(
                    "terminated",
                    False,
                ),
            )
        )
        for item in items
    )

    truncated = sum(
        bool(
            item.get(
                "truncated",
                False,
            )
        )
        for item in items
    )

    failed = len(items) - successful

    return EpisodeSummary(
        total_episodes=len(items),
        successful_episodes=successful,
        failed_episodes=failed,
        truncated_episodes=truncated,
        mean_reward=_mean(rewards),
        total_reward=sum(rewards),
        mean_episode_length=_mean(lengths),
    )


def build_performance_metrics(
    summary: EpisodeSummary,
    *,
    evaluation_time_seconds: float = 0.0,
) -> PerformanceMetrics:

    episodes_per_second = (
        summary.total_episodes
        / evaluation_time_seconds
        if evaluation_time_seconds > 0
        else 0.0
    )

    return PerformanceMetrics(
        success_rate=summary.success_rate,
        mean_reward=summary.mean_reward,
        mean_episode_length=summary.mean_episode_length,
        evaluation_time_seconds=evaluation_time_seconds,
        episodes_per_second=episodes_per_second,
    )


def build_performance_from_rewards(
    rewards: Iterable[float],
    successes: Iterable[bool],
    lengths: Iterable[float] = (),
    *,
    evaluation_time_seconds: float = 0.0,
) -> PerformanceMetrics:

    reward_values = [
        float(value)
        for value in rewards
    ]

    success_values = [
        bool(value)
        for value in successes
    ]

    length_values = [
        float(value)
        for value in lengths
    ]

    count = len(success_values)

    success_rate = (
        sum(success_values) / count
        if count
        else 0.0
    )

    return PerformanceMetrics(
        success_rate=success_rate,
        mean_reward=_mean(reward_values),
        mean_episode_length=_mean(length_values),
        reward_std=_population_std(reward_values),
        evaluation_time_seconds=evaluation_time_seconds,
        episodes_per_second=(
            count / evaluation_time_seconds
            if evaluation_time_seconds > 0
            else 0.0
        ),
    )
