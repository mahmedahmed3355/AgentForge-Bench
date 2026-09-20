from agentforge.integration.pipeline import (
    IntegrationPipeline,
    IntegrationResult,
)


def test_pipeline_accepts_valid_runner():
    expected = IntegrationResult(
        success=True,
        steps=3,
        total_reward=2.5,
        terminated=True,
        truncated=False,
        metadata={"source": "test"},
    )

    pipeline = IntegrationPipeline(lambda: expected)

    assert pipeline.run() == expected
