from evaluator.hidden_scenarios import get_hidden_scenarios
from evaluator.harness import run_hidden_evaluation, run_hidden_scenario


def test_hidden_scenarios_are_unseen_and_unique():
    scenarios = get_hidden_scenarios()

    assert len(scenarios) >= 4
    assert len({item.scenario_id for item in scenarios}) == len(scenarios)
    assert len({item.seed for item in scenarios}) == len(scenarios)

    public_ids = {
        "decision-v1-public-001",
        "decision-v1-public-002",
        "decision-v1-public-003",
    }

    assert not any(item.scenario_id in public_ids for item in scenarios)


def test_each_hidden_scenario_passes_oracle_and_verifier():
    for scenario in get_hidden_scenarios():
        result = run_hidden_scenario(scenario)

        assert result["oracle_solved"] is True, result
        assert result["verifier_verified"] is True, result
        assert result["terminal"] is True, result
        assert result["success"] is True, result
        assert result["failure_detected"] is False, result
        assert result["recovery_required"] is False, result
        assert result["replanning_required"] is False, result
        assert result["validation_status"] == "globally_validated", result


def test_hidden_evaluation_passes():
    report = run_hidden_evaluation()

    assert report["num_scenarios"] == 4
    assert report["oracle_pass"] is True
    assert report["verifier_pass"] is True
    assert report["hidden_evaluation_pass"] is True
