import random

from agentforge.scenario.generator import DemoScenarioGenerator


def test_generation_is_deterministic_for_same_seed() -> None:
    generator = DemoScenarioGenerator("demo")

    first = generator.generate(
        scenario_id="scenario-001",
        seed=42,
    )
    second = generator.generate(
        scenario_id="scenario-001",
        seed=42,
    )

    assert first.public == second.public
    assert first.hidden == second.hidden
    assert first.metadata == second.metadata
    assert first.identity == second.identity


def test_different_seeds_can_produce_different_data() -> None:
    generator = DemoScenarioGenerator("demo")

    first = generator.generate(
        scenario_id="scenario-001",
        seed=1,
    )
    second = generator.generate(
        scenario_id="scenario-001",
        seed=2,
    )

    assert first.public != second.public or first.hidden != second.hidden


def test_generation_does_not_modify_global_rng() -> None:
    generator = DemoScenarioGenerator("demo")

    random.seed(12345)
    before = random.getstate()

    generator.generate(
        scenario_id="scenario-001",
        seed=99,
    )

    after = random.getstate()

    assert before == after
