from __future__ import annotations

from dataclasses import dataclass

from data_pipeline_recovery_v1.decision_graph import IncidentKind


@dataclass(frozen=True)
class HiddenScenario:
    scenario_id: str
    seed: int
    incident_family: IncidentKind


HIDDEN_SCENARIOS = (
    HiddenScenario(
        scenario_id="pipeline-v1-hidden-001",
        seed=9101,
        incident_family=IncidentKind.RESOURCE,
    ),
    HiddenScenario(
        scenario_id="pipeline-v1-hidden-002",
        seed=9102,
        incident_family=IncidentKind.SCHEMA,
    ),
    HiddenScenario(
        scenario_id="pipeline-v1-hidden-003",
        seed=9103,
        incident_family=IncidentKind.TRANSFORMATION,
    ),
)


def load_hidden_scenarios() -> tuple[HiddenScenario, ...]:
    return HIDDEN_SCENARIOS
