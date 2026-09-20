
from agentforge.reporting import (
    TaskLevelResult,
    analyze_capabilities,
    analyze_failures,
    analyze_generalization,
    analyze_scenarios,
    analyze_trajectories,
    compare_oracle,
    summarize_verifier,
)


def results():
    return [
        TaskLevelResult(
            "a",
            "passed",
            1,
            1.0,
            10.0,
            oracle_score=10.0,
            failures=(),
            metadata={"capability": "cuda"},
        ),
        TaskLevelResult(
            "b",
            "failed",
            1,
            0.0,
            2.0,
            oracle_score=4.0,
            failures=("timeout",),
            metadata={"capability": "cuda"},
        ),
    ]


def test_capability_analysis():
    report = analyze_capabilities(
        results()
    )

    assert (
        report.capabilities["cuda"]
        == 0.5
    )


def test_failure_analysis():
    report = analyze_failures(
        results()
    )

    assert report.failure_counts["timeout"] == 1
    assert report.examples["timeout"] == ("b",)


def test_trajectory_analysis():
    report = analyze_trajectories(
        [
            {"valid": True},
            {"valid": False},
        ]
    )

    assert report.total_trajectories == 2
    assert report.valid_trajectories == 1


def test_scenario_analysis():
    report = analyze_scenarios(
        [
            {"scenario_id": "s1"},
            {
                "scenario_id": "s1",
                "hidden": True,
            },
        ]
    )

    assert report.unique_scenario_ids == 1
    assert report.public_scenarios == 1
    assert report.hidden_scenarios == 1


def test_oracle_comparison():
    report = compare_oracle(
        results()
    )

    assert report.compared_tasks == 2
    assert report.exact_matches == 1


def test_verifier_summary():
    report = summarize_verifier(
        [
            {"passed": True},
            {
                "passed": False,
                "diagnostic": "bad reward",
            },
        ]
    )

    assert report.pass_rate == 0.5
    assert report.diagnostics == (
        "bad reward",
    )


def test_generalization():
    report = analyze_generalization(
        0.8,
        0.5,
    )

    assert (
        abs(
            report.generalization_gap - 0.3
        )
        < 1e-12
    )
