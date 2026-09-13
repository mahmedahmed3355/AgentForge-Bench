from __future__ import annotations

from dataclasses import dataclass

from branching_decision_recovery_v1.scenario_generator import (
    ScenarioConfig,
    ScenarioGenerator,
)


@dataclass(frozen=True)
class HiddenScenario:
    scenario_id: str
    seed: int
    scenario_family: str
    config: ScenarioConfig


_HIDDEN_SEEDS = (
    9301,
    9302,
    9303,
    9304,
)


def get_hidden_scenarios() -> tuple[HiddenScenario, ...]:
    generator = ScenarioGenerator()
    scenarios: list[HiddenScenario] = []

    families = (
        "dependency",
        "configuration",
        "cascade",
        "dependency",
    )

    for index, (seed, family) in enumerate(
        zip(_HIDDEN_SEEDS, families),
        start=1,
    ):
        scenario_id = f"decision-v1-hidden-{index:03d}"

        config = generator.generate(
            seed=seed,
            scenario_id=scenario_id,
            scenario_family=family,
        )

        scenarios.append(
            HiddenScenario(
                scenario_id=scenario_id,
                seed=seed,
                scenario_family=family,
                config=config,
            )
        )

    return tuple(scenarios)
