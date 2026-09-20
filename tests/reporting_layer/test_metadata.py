from agentforge.reporting.metadata import ReportMetadata


def test_metadata_as_dict():
    report = ReportMetadata("AgentForge-RL-Bench", seed=42)
    data = report.as_dict()
    assert data["benchmark_name"] == "AgentForge-RL-Bench"
    assert data["seed"] == 42
