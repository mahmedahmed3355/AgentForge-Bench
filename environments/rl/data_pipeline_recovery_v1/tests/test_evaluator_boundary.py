from __future__ import annotations

from pathlib import Path

from evaluator.hidden_scenarios import load_hidden_scenarios
from evaluator.harness import run_hidden_evaluation


def test_hidden_scenarios_are_outside_public_package():
    package_root = Path(__file__).resolve().parents[1] / "data_pipeline_recovery_v1"
    hidden_path = Path(__file__).resolve().parents[1] / "evaluator" / "hidden_scenarios.py"

    assert hidden_path.exists()
    assert not (package_root / "hidden_scenarios.py").exists()


def test_hidden_scenarios_are_not_public_scenarios():
    public_path = Path(__file__).resolve().parents[1] / "data" / "scenarios.json"
    public_text = public_path.read_text()

    for scenario in load_hidden_scenarios():
        assert scenario.scenario_id not in public_text


def test_hidden_evaluation_reference_passes():
    results = run_hidden_evaluation()

    assert len(results) == 3
    assert all(result.oracle_success for result in results)
    assert all(result.verifier_success for result in results)
    assert all(result.terminal for result in results)
