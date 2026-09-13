from branching_decision_recovery_v1.scenario_generator import (
    ScenarioGenerator,
    generate_hidden_scenarios,
)


def test_scenario_generation_is_deterministic():
    generator = ScenarioGenerator()

    first = generator.generate(seed=9301)
    second = generator.generate(seed=9301)

    assert first == second


def test_different_seeds_produce_structural_variation():
    generator = ScenarioGenerator()

    first = generator.generate(seed=9301)
    second = generator.generate(seed=9302)

    assert (
        first.topology != second.topology
        or first.symptom_profile != second.symptom_profile
        or first.branch_costs != second.branch_costs
        or first.branch_risks != second.branch_risks
        or first.consequence_delay != second.consequence_delay
    )


def test_generated_scenario_has_nontrivial_dependencies():
    generator = ScenarioGenerator()

    scenario = generator.generate(seed=9301)

    assert len(scenario.topology) >= 4
    assert len(scenario.dependencies) >= 3
    assert all(len(edge) == 2 for edge in scenario.dependencies)


def test_generated_costs_and_risks_are_bounded():
    generator = ScenarioGenerator()

    scenario = generator.generate(seed=9301)

    assert all(5 <= value <= 18 for value in scenario.branch_costs)
    assert all(1 <= value <= 5 for value in scenario.branch_risks)
    assert all(3 <= value <= 12 for value in scenario.strategy_costs)
    assert all(1 <= value <= 5 for value in scenario.strategy_risks)
    assert 3 <= scenario.consequence_delay <= 8


def test_hidden_scenarios_are_unseen_and_unique():
    scenarios = generate_hidden_scenarios(
        (9301, 9302, 9303, 9304)
    )

    assert len(scenarios) == 4
    assert len({item.scenario_id for item in scenarios}) == 4
    assert len({item.seed for item in scenarios}) == 4

    public_ids = {
        "decision-v1-public-001",
        "decision-v1-public-002",
        "decision-v1-public-003",
    }

    assert not any(
        item.scenario_id in public_ids
        for item in scenarios
    )
