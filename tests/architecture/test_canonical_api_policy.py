from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LEGACY_MODULES = (
    "src/agentforge/adapter/agent.py",
    "src/agentforge/adapter/gymnasium.py",
    "src/agentforge/adapter/agentforge_bench.py",
    "src/agentforge/integration/episode.py",
    "src/agentforge/integration/evaluation.py",
    "src/agentforge/orchestration/episode.py",
    "src/agentforge/orchestration/evaluation.py",
    "src/agentforge/oracle/trajectory.py",
    "src/agentforge/reporting/episode_summary.py",
    "src/agentforge/reporting/trajectory_analysis.py",
    "src/agentforge/reporting/verifier_results.py",
)


def test_legacy_modules_exist():
    assert all((ROOT / path).is_file() for path in LEGACY_MODULES)


def test_legacy_policy_is_documented():
    text = (ROOT / "docs/architecture.md").read_text()
    assert "## Legacy Module Policy" in text
    assert "compatibility modules" in text
    assert "not canonical APIs" in text


def test_every_legacy_module_is_documented():
    text = (ROOT / "docs/architecture.md").read_text()
    assert all(path in text for path in LEGACY_MODULES)
