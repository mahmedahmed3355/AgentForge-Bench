from agentforge.reporting.scenario_analysis import ScenarioAnalysis


def test_scenario_analysis():
    analysis = ScenarioAnalysis(["a", "b", "a"])
    assert analysis.count == 3
    assert analysis.unique_count == 2
