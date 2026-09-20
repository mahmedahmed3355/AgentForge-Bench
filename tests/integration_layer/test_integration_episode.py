from agentforge.integration.episode import run_integrated_episode
from agentforge.integration.pipeline import IntegrationPipeline, IntegrationResult


def test_integrated_episode_accumulates_results():
    result = IntegrationResult(
        success=True,
        steps=2,
        total_reward=1.5,
        terminated=True,
        truncated=False,
        metadata={},
    )

    pipeline = IntegrationPipeline(lambda: result)
    episode = run_integrated_episode(pipeline, episodes=3)

    assert len(episode.results) == 3
    assert episode.steps == 6
    assert episode.total_reward == 4.5
    assert episode.success is True
