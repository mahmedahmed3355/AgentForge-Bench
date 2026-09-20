
import json

from agentforge.reporting import (
    BenchmarkMetadata,
    BenchmarkReport,
    EpisodeSummary,
    FinalSummary,
    PerformanceMetrics,
    TaskCoverage,
    TaskLevelResult,
)


def make_report():
    return BenchmarkReport(
        metadata=BenchmarkMetadata(
            "af",
            "AgentForge",
            "1.0",
        ),
        task_coverage=TaskCoverage(
            2,
            2,
            1,
            1,
        ),
        episode_summary=EpisodeSummary(
            4,
            2,
            2,
            mean_reward=0.5,
        ),
        performance_metrics=PerformanceMetrics(
            0.5,
            0.5,
            3.0,
        ),
        task_level_results=(
            TaskLevelResult(
                "t1",
                "passed",
                2,
                1.0,
                1.0,
            ),
        ),
        final_summary=FinalSummary(
            "complete",
            "done",
        ),
    )


def test_report_round_trip():
    report = make_report()

    restored = BenchmarkReport.from_dict(
        report.to_dict()
    )

    assert restored == report


def test_report_json_is_valid():
    payload = json.loads(
        make_report().to_json()
    )

    assert payload["metadata"]["benchmark_id"] == "af"


def test_coverage_rate():
    assert (
        TaskCoverage(
            4,
            2,
            1,
            1,
        ).coverage_rate
        == 0.5
    )


def test_episode_success_rate():
    assert (
        EpisodeSummary(
            4,
            3,
            1,
        ).success_rate
        == 0.75
    )
