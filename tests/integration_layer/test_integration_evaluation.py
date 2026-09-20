from agentforge.integration.evaluation import evaluate_integrated
from agentforge.integration.pipeline import IntegrationPipeline, IntegrationResult


def test_integrated_evaluation_reports_metrics():
    result = IntegrationResult(
        success=True,
        steps=4,
        total_reward=3.0,
        terminated=True,
        truncated=False,
        metadata={"verified": True},
    )

    pipeline = IntegrationPipeline(lambda: result)
    evaluation = evaluate_integrated(pipeline, episodes=2)

    assert evaluation.episodes == 2
    assert evaluation.successful_episodes == 2
    assert evaluation.total_steps == 8
    assert evaluation.total_reward == 6.0
    assert evaluation.success_rate == 1.0
