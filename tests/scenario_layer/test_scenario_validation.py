import pytest

from agentforge.scenario.base import Scenario
from agentforge.scenario.identity import ScenarioIdentity
from agentforge.scenario.validation import (
    ScenarioValidationError,
    validate_scenario,
)


def test_valid_scenario_passes_validation() -> None:
    scenario = Scenario(
        identity=ScenarioIdentity("demo", "scenario-001", 7),
        public={"value": 1},
        hidden={"target": 2},
        metadata={"version": 1},
    )

    validate_scenario(scenario)


def test_invalid_object_is_rejected() -> None:
    with pytest.raises(ScenarioValidationError):
        validate_scenario(object())


def test_agent_view_does_not_expose_hidden_data() -> None:
    scenario = Scenario(
        identity=ScenarioIdentity("demo", "scenario-001", 7),
        public={"value": 1},
        hidden={"answer": "secret"},
    )

    view = scenario.agent_view()

    assert view == {"value": 1}
    assert "answer" not in view
