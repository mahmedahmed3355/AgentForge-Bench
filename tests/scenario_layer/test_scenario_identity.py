from agentforge.scenario.identity import ScenarioIdentity


def test_identity_is_stable() -> None:
    identity = ScenarioIdentity(
        task_id="demo",
        scenario_id="scenario-001",
        seed=7,
    )

    assert identity.task_id == "demo"
    assert identity.scenario_id == "scenario-001"
    assert identity.seed == 7
    assert identity.key == "demo:scenario-001:7"


def test_identity_is_hashable() -> None:
    identity = ScenarioIdentity(
        task_id="demo",
        scenario_id="scenario-001",
        seed=7,
    )

    assert hash(identity) == hash(identity)
