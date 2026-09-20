from agentforge.scenario.base import Scenario
from agentforge.scenario.identity import ScenarioIdentity


def make_scenario() -> Scenario:
    return Scenario(
        identity=ScenarioIdentity(
            task_id="demo",
            scenario_id="scenario-001",
            seed=11,
        ),
        public={"visible": 1},
        hidden={"secret": 2},
        metadata={"source": "test"},
    )


def test_scenario_preserves_identity() -> None:
    scenario = make_scenario()

    assert scenario.task_id == "demo"
    assert scenario.scenario_id == "scenario-001"
    assert scenario.seed == 11


def test_agent_view_exposes_public_data_only() -> None:
    scenario = make_scenario()

    view = scenario.agent_view()

    assert view == {"visible": 1}
    assert "secret" not in view


def test_scenario_copies_input_mappings() -> None:
    public = {"value": 1}

    scenario = Scenario(
        identity=ScenarioIdentity("demo", "scenario-001", 1),
        public=public,
    )

    public["value"] = 99

    assert scenario.public["value"] == 1
