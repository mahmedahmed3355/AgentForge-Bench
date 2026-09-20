
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchmarkMetadata:
    benchmark_id: str
    name: str
    version: str
    framework_version: str = "unknown"
    generated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must not be empty")

        if not self.name:
            raise ValueError("name must not be empty")

        if not self.version:
            raise ValueError("version must not be empty")

        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                "1970-01-01T00:00:00+00:00",
            )


@dataclass(frozen=True)
class TaskCoverage:
    total_tasks: int
    evaluated_tasks: int
    passed_tasks: int
    failed_tasks: int
    skipped_tasks: int = 0

    def __post_init__(self) -> None:
        values = (
            self.total_tasks,
            self.evaluated_tasks,
            self.passed_tasks,
            self.failed_tasks,
            self.skipped_tasks,
        )

        if any(value < 0 for value in values):
            raise ValueError("coverage counts must be non-negative")

        if self.evaluated_tasks > self.total_tasks:
            raise ValueError(
                "evaluated_tasks cannot exceed total_tasks"
            )

        if self.passed_tasks + self.failed_tasks > self.evaluated_tasks:
            raise ValueError(
                "passed_tasks + failed_tasks cannot exceed evaluated_tasks"
            )

    @property
    def coverage_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0

        return self.evaluated_tasks / self.total_tasks


@dataclass(frozen=True)
class EpisodeSummary:
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    truncated_episodes: int = 0
    mean_reward: float = 0.0
    total_reward: float = 0.0
    mean_episode_length: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.total_episodes,
            self.successful_episodes,
            self.failed_episodes,
            self.truncated_episodes,
        )

        if any(value < 0 for value in values):
            raise ValueError("episode counts must be non-negative")

        if (
            self.successful_episodes + self.failed_episodes
            > self.total_episodes
        ):
            raise ValueError(
                "successful_episodes + failed_episodes "
                "cannot exceed total_episodes"
            )

    @property
    def success_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0

        return self.successful_episodes / self.total_episodes


@dataclass(frozen=True)
class PerformanceMetrics:
    success_rate: float
    mean_reward: float
    mean_episode_length: float
    reward_std: float = 0.0
    evaluation_time_seconds: float = 0.0
    episodes_per_second: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError(
                "success_rate must be between 0 and 1"
            )

        if self.reward_std < 0:
            raise ValueError("reward_std must be non-negative")

        if self.evaluation_time_seconds < 0:
            raise ValueError(
                "evaluation_time_seconds must be non-negative"
            )

        if self.episodes_per_second < 0:
            raise ValueError(
                "episodes_per_second must be non-negative"
            )


@dataclass(frozen=True)
class TaskLevelResult:
    task_id: str
    status: str
    episodes: int
    success_rate: float
    mean_reward: float
    oracle_score: float | None = None
    verifier_pass_rate: float | None = None
    failures: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")

        if self.episodes < 0:
            raise ValueError("episodes must be non-negative")

        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError(
                "success_rate must be between 0 and 1"
            )

        if self.oracle_score is not None and self.oracle_score < 0:
            raise ValueError(
                "oracle_score must be non-negative"
            )

        if (
            self.verifier_pass_rate is not None
            and not 0.0 <= self.verifier_pass_rate <= 1.0
        ):
            raise ValueError(
                "verifier_pass_rate must be between 0 and 1"
            )


@dataclass(frozen=True)
class CapabilityAnalysis:
    capabilities: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FailureAnalysis:
    failure_counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, tuple[str, ...]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryAnalysis:
    total_trajectories: int = 0
    valid_trajectories: int = 0
    invalid_trajectories: int = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioAnalysis:
    total_scenarios: int = 0
    public_scenarios: int = 0
    hidden_scenarios: int = 0
    unique_scenario_ids: int = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OracleComparison:
    compared_tasks: int = 0
    exact_matches: int = 0
    mean_score_ratio: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifierResults:
    checked_episodes: int = 0
    passed_episodes: int = 0
    failed_episodes: int = 0
    diagnostics: tuple[str, ...] = ()

    @property
    def pass_rate(self) -> float:
        if self.checked_episodes == 0:
            return 0.0

        return self.passed_episodes / self.checked_episodes


@dataclass(frozen=True)
class GeneralizationAnalysis:
    seen_scenario_success_rate: float = 0.0
    unseen_scenario_success_rate: float = 0.0
    generalization_gap: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinalSummary:
    overall_status: str
    headline: str
    key_findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkReport:
    metadata: BenchmarkMetadata
    task_coverage: TaskCoverage
    episode_summary: EpisodeSummary
    performance_metrics: PerformanceMetrics
    task_level_results: tuple[TaskLevelResult, ...] = ()
    capability_analysis: CapabilityAnalysis = field(
        default_factory=CapabilityAnalysis
    )
    failure_analysis: FailureAnalysis = field(
        default_factory=FailureAnalysis
    )
    trajectory_analysis: TrajectoryAnalysis = field(
        default_factory=TrajectoryAnalysis
    )
    scenario_analysis: ScenarioAnalysis = field(
        default_factory=ScenarioAnalysis
    )
    oracle_comparison: OracleComparison = field(
        default_factory=OracleComparison
    )
    verifier_results: VerifierResults = field(
        default_factory=VerifierResults
    )
    generalization_analysis: GeneralizationAnalysis = field(
        default_factory=GeneralizationAnalysis
    )
    final_summary: FinalSummary = field(
        default_factory=lambda: FinalSummary(
            "unknown",
            "",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "BenchmarkReport":

        def tuple_value(
            mapping: Mapping[str, Any],
            key: str,
        ) -> tuple[str, ...]:
            return tuple(mapping.get(key, ()))

        metadata_data = data["metadata"]

        coverage_data = data["task_coverage"]

        episode_data = data["episode_summary"]

        performance_data = data["performance_metrics"]

        capability_data = data.get(
            "capability_analysis",
            {},
        )

        failure_data = data.get(
            "failure_analysis",
            {},
        )

        trajectory_data = data.get(
            "trajectory_analysis",
            {},
        )

        scenario_data = data.get(
            "scenario_analysis",
            {},
        )

        oracle_data = data.get(
            "oracle_comparison",
            {},
        )

        verifier_data = data.get(
            "verifier_results",
            {},
        )

        generalization_data = data.get(
            "generalization_analysis",
            {},
        )

        summary_data = data.get(
            "final_summary",
            {
                "overall_status": "unknown",
                "headline": "",
            },
        )

        task_results = tuple(
            TaskLevelResult(
                **{
                    key: value
                    for key, value in dict(item).items()
                    if key != "failures"
                },
                failures=tuple(item.get("failures", ())),
            )
            for item in data.get(
                "task_level_results",
                (),
            )
        )

        return cls(
            metadata=BenchmarkMetadata(
                **dict(metadata_data),
            ),
            task_coverage=TaskCoverage(
                **dict(coverage_data),
            ),
            episode_summary=EpisodeSummary(
                **dict(episode_data),
            ),
            performance_metrics=PerformanceMetrics(
                **dict(performance_data),
            ),
            task_level_results=task_results,
            capability_analysis=CapabilityAnalysis(
                capabilities=dict(
                    capability_data.get(
                        "capabilities",
                        {},
                    )
                ),
                notes=tuple_value(
                    capability_data,
                    "notes",
                ),
            ),
            failure_analysis=FailureAnalysis(
                failure_counts=dict(
                    failure_data.get(
                        "failure_counts",
                        {},
                    )
                ),
                examples={
                    key: tuple(value)
                    for key, value in failure_data.get(
                        "examples",
                        {},
                    ).items()
                },
                notes=tuple_value(
                    failure_data,
                    "notes",
                ),
            ),
            trajectory_analysis=TrajectoryAnalysis(
                total_trajectories=trajectory_data.get(
                    "total_trajectories",
                    0,
                ),
                valid_trajectories=trajectory_data.get(
                    "valid_trajectories",
                    0,
                ),
                invalid_trajectories=trajectory_data.get(
                    "invalid_trajectories",
                    0,
                ),
                notes=tuple_value(
                    trajectory_data,
                    "notes",
                ),
            ),
            scenario_analysis=ScenarioAnalysis(
                total_scenarios=scenario_data.get(
                    "total_scenarios",
                    0,
                ),
                public_scenarios=scenario_data.get(
                    "public_scenarios",
                    0,
                ),
                hidden_scenarios=scenario_data.get(
                    "hidden_scenarios",
                    0,
                ),
                unique_scenario_ids=scenario_data.get(
                    "unique_scenario_ids",
                    0,
                ),
                notes=tuple_value(
                    scenario_data,
                    "notes",
                ),
            ),
            oracle_comparison=OracleComparison(
                compared_tasks=oracle_data.get(
                    "compared_tasks",
                    0,
                ),
                exact_matches=oracle_data.get(
                    "exact_matches",
                    0,
                ),
                mean_score_ratio=oracle_data.get(
                    "mean_score_ratio",
                    0.0,
                ),
                notes=tuple_value(
                    oracle_data,
                    "notes",
                ),
            ),
            verifier_results=VerifierResults(
                checked_episodes=verifier_data.get(
                    "checked_episodes",
                    0,
                ),
                passed_episodes=verifier_data.get(
                    "passed_episodes",
                    0,
                ),
                failed_episodes=verifier_data.get(
                    "failed_episodes",
                    0,
                ),
                diagnostics=tuple_value(
                    verifier_data,
                    "diagnostics",
                ),
            ),
            generalization_analysis=GeneralizationAnalysis(
                seen_scenario_success_rate=generalization_data.get(
                    "seen_scenario_success_rate",
                    0.0,
                ),
                unseen_scenario_success_rate=generalization_data.get(
                    "unseen_scenario_success_rate",
                    0.0,
                ),
                generalization_gap=generalization_data.get(
                    "generalization_gap",
                    0.0,
                ),
                notes=tuple_value(
                    generalization_data,
                    "notes",
                ),
            ),
            final_summary=FinalSummary(
                overall_status=summary_data.get(
                    "overall_status",
                    "unknown",
                ),
                headline=summary_data.get(
                    "headline",
                    "",
                ),
                key_findings=tuple_value(
                    summary_data,
                    "key_findings",
                ),
            ),
        )
