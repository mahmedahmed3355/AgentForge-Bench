from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    seed: int
    scenario_family: str
    topology: tuple[str, ...]
    dependencies: tuple[tuple[str, str], ...]
    symptom_profile: str
    branch_costs: tuple[float, float, float]
    branch_risks: tuple[float, float, float]
    strategy_costs: tuple[float, float, float]
    strategy_risks: tuple[float, float, float]
    consequence_delay: int


class ScenarioGenerator:
    """Deterministic scenario generator for unseen evaluation cases."""

    FAMILIES = (
        "dependency",
        "configuration",
        "cascade",
    )

    TOPOLOGIES = (
        "linear",
        "fanout",
        "diamond",
        "layered",
    )

    SYMPTOMS = (
        "latency_spike",
        "validation_drift",
        "dependency_mismatch",
        "resource_pressure",
    )

    def generate(
        self,
        seed: int,
        scenario_id: str | None = None,
        scenario_family: str | None = None,
    ) -> ScenarioConfig:
        rng = Random(seed)

        family = (
            scenario_family
            if scenario_family in self.FAMILIES
            else self.FAMILIES[rng.randrange(len(self.FAMILIES))]
        )

        topology = self.TOPOLOGIES[rng.randrange(len(self.TOPOLOGIES))]
        symptom = self.SYMPTOMS[rng.randrange(len(self.SYMPTOMS))]

        if topology == "linear":
            nodes = ("source", "processing", "validation", "output")
            dependencies = (
                ("source", "processing"),
                ("processing", "validation"),
                ("validation", "output"),
            )
        elif topology == "fanout":
            nodes = (
                "source",
                "processing_a",
                "processing_b",
                "validation",
                "output",
            )
            dependencies = (
                ("source", "processing_a"),
                ("source", "processing_b"),
                ("processing_a", "validation"),
                ("processing_b", "validation"),
                ("validation", "output"),
            )
        elif topology == "diamond":
            nodes = (
                "source",
                "processing_a",
                "processing_b",
                "merge",
                "validation",
                "output",
            )
            dependencies = (
                ("source", "processing_a"),
                ("source", "processing_b"),
                ("processing_a", "merge"),
                ("processing_b", "merge"),
                ("merge", "validation"),
                ("validation", "output"),
            )
        else:
            nodes = (
                "source",
                "processing",
                "dependency",
                "validation",
                "output",
            )
            dependencies = (
                ("source", "processing"),
                ("processing", "dependency"),
                ("dependency", "validation"),
                ("validation", "output"),
            )

        branch_costs = tuple(
            float(rng.randint(5, 18)) for _ in range(3)
        )
        branch_risks = tuple(
            float(rng.randint(1, 5)) for _ in range(3)
        )
        strategy_costs = tuple(
            float(rng.randint(3, 12)) for _ in range(3)
        )
        strategy_risks = tuple(
            float(rng.randint(1, 5)) for _ in range(3)
        )

        consequence_delay = rng.randint(3, 8)

        return ScenarioConfig(
            scenario_id=scenario_id or f"generated-{seed}",
            seed=seed,
            scenario_family=family,
            topology=topology,
            dependencies=dependencies,
            symptom_profile=symptom,
            branch_costs=branch_costs,
            branch_risks=branch_risks,
            strategy_costs=strategy_costs,
            strategy_risks=strategy_risks,
            consequence_delay=consequence_delay,
        )


def generate_hidden_scenarios(
    seeds: tuple[int, ...],
) -> tuple[ScenarioConfig, ...]:
    generator = ScenarioGenerator()

    return tuple(
        generator.generate(
            seed=seed,
            scenario_id=f"decision-v1-hidden-{index:03d}",
        )
        for index, seed in enumerate(seeds, start=1)
    )
