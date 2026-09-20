from agentforge.runtime.episode_report import (
    build_episode_report,
)
from agentforge.runtime.episode_runner import EpisodeResult


def test_episode_report_preserves_result_data():
    result = EpisodeResult(
        observation="final",
        total_reward=4.0,
        step_count=4,
        terminated=True,
        truncated=False,
        trajectory=("a", "b"),
    )

    report = build_episode_report(
        result,
        metadata={"task": "demo"},
    )

    assert report.step_count == 4
    assert report.total_reward == 4.0
    assert report.terminated is True
    assert report.truncated is False
    assert report.metadata == {"task": "demo"}
