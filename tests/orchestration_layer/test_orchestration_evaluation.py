from agentforge.orchestration import (
    EpisodeResult,
    summarize,
)


def test_evaluation_summary() -> None:
    results = [
        EpisodeResult(
            task_id="t",
            scenario_id="s",
            steps=3,
            total_reward=3.0,
            terminated=True,
            truncated=False,
            success=True,
        ),
        EpisodeResult(
            task_id="t",
            scenario_id="s",
            steps=2,
            total_reward=1.0,
            terminated=True,
            truncated=False,
            success=False,
        ),
    ]

    summary = summarize(results)

    assert summary.episodes == 2
    assert summary.successful_episodes == 1
    assert summary.success_rate == 0.5
    assert summary.mean_reward == 2.0
    assert summary.total_steps == 5


def test_empty_evaluation_summary() -> None:
    summary = summarize([])

    assert summary.episodes == 0
    assert summary.success_rate == 0.0
    assert summary.mean_reward == 0.0
