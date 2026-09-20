
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .benchmark_report import (
    CapabilityAnalysis,
    FailureAnalysis,
    GeneralizationAnalysis,
    OracleComparison,
    ScenarioAnalysis,
    TaskLevelResult,
    TrajectoryAnalysis,
    VerifierResults,
)


def analyze_capabilities(
    task_results: Iterable[TaskLevelResult],
    capability_key: str = "capability",
) -> CapabilityAnalysis:

    grouped: dict[str, list[float]] = {}

    for result in task_results:
        capability = str(
            result.metadata.get(
                capability_key,
                "unknown",
            )
        )

        grouped.setdefault(
            capability,
            [],
        ).append(
            result.success_rate
        )

    scores = {
        name: sum(values) / len(values)
        for name, values in grouped.items()
        if values
    }

    return CapabilityAnalysis(
        capabilities=scores
    )


def analyze_failures(
    task_results: Iterable[TaskLevelResult],
) -> FailureAnalysis:

    counts: Counter[str] = Counter()

    examples: dict[str, list[str]] = {}

    for result in task_results:
        for failure in result.failures:
            counts[failure] += 1

            examples.setdefault(
                failure,
                [],
            ).append(
                result.task_id
            )

    return FailureAnalysis(
        failure_counts=dict(counts),
        examples={
            key: tuple(value)
            for key, value in examples.items()
        },
    )


def analyze_trajectories(
    trajectories: Iterable[Mapping[str, Any]],
) -> TrajectoryAnalysis:

    items = list(trajectories)

    valid = sum(
        bool(
            item.get(
                "valid",
                False,
            )
        )
        for item in items
    )

    return TrajectoryAnalysis(
        total_trajectories=len(items),
        valid_trajectories=valid,
        invalid_trajectories=len(items) - valid,
    )


def analyze_scenarios(
    scenarios: Iterable[Mapping[str, Any]],
) -> ScenarioAnalysis:

    items = list(scenarios)

    scenario_ids = {
        str(item["scenario_id"])
        for item in items
        if "scenario_id" in item
    }

    public = sum(
        not bool(
            item.get(
                "hidden",
                False,
            )
        )
        for item in items
    )

    hidden = len(items) - public

    return ScenarioAnalysis(
        total_scenarios=len(items),
        public_scenarios=public,
        hidden_scenarios=hidden,
        unique_scenario_ids=len(scenario_ids),
    )


def compare_oracle(
    task_results: Iterable[TaskLevelResult],
    *,
    exact_tolerance: float = 0.0,
) -> OracleComparison:

    items = [
        item
        for item in task_results
        if item.oracle_score is not None
    ]

    if not items:
        return OracleComparison()

    exact = sum(
        abs(
            item.mean_reward
            - float(item.oracle_score)
        ) <= exact_tolerance
        for item in items
    )

    ratios = [
        item.mean_reward
        / float(item.oracle_score)
        for item in items
        if float(item.oracle_score) != 0.0
    ]

    return OracleComparison(
        compared_tasks=len(items),
        exact_matches=exact,
        mean_score_ratio=(
            sum(ratios) / len(ratios)
            if ratios
            else 0.0
        ),
    )


def summarize_verifier(
    checked: Iterable[Mapping[str, Any]],
) -> VerifierResults:

    items = list(checked)

    passed = sum(
        bool(
            item.get(
                "passed",
                False,
            )
        )
        for item in items
    )

    diagnostics = tuple(
        str(item["diagnostic"])
        for item in items
        if item.get("diagnostic")
    )

    return VerifierResults(
        checked_episodes=len(items),
        passed_episodes=passed,
        failed_episodes=len(items) - passed,
        diagnostics=diagnostics,
    )


def analyze_generalization(
    seen_success_rate: float,
    unseen_success_rate: float,
) -> GeneralizationAnalysis:

    if not 0.0 <= seen_success_rate <= 1.0:
        raise ValueError(
            "seen_success_rate must be between 0 and 1"
        )

    if not 0.0 <= unseen_success_rate <= 1.0:
        raise ValueError(
            "unseen_success_rate must be between 0 and 1"
        )

    return GeneralizationAnalysis(
        seen_scenario_success_rate=seen_success_rate,
        unseen_scenario_success_rate=unseen_success_rate,
        generalization_gap=(
            seen_success_rate
            - unseen_success_rate
        ),
    )
