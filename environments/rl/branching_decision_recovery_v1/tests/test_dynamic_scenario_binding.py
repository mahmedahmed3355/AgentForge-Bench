from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv
from branching_decision_recovery_v1.scenario_generator import ScenarioGenerator


def test_scenario_config_changes_branch_cost_and_risk():
    generator = ScenarioGenerator()
    config = generator.generate(
        seed=9901,
        scenario_id="dynamic-test-001",
        scenario_family="dependency",
    )

    env = BranchingDecisionRecoveryEnv()
    env.scenario_config = config
    env.reset(seed=config.seed)

    env.step("inspect_system")
    env.step("inspect_component", component="processing")
    env.step("inspect_dependency", dependency="processing")
    env.step("inspect_history")
    env.step(
        "form_hypothesis",
        hypothesis="downstream_dependency_risk",
        confidence=0.8,
    )
    env.step("select_branch", branch="branch_a")
    env.step("select_strategy", strategy="strategy_a")

    _, reward, _, _ = env.step("apply_action")

    assert reward == 1.0
    assert env.state is not None
    assert env.state.decision_cost == config.branch_costs[0]
    assert env.state.risk_level == config.branch_risks[0]


def test_generated_topology_is_bound_to_dependency_state():
    generator = ScenarioGenerator()
    config = generator.generate(
        seed=9902,
        scenario_id="dynamic-test-002",
        scenario_family="cascade",
    )

    env = BranchingDecisionRecoveryEnv()
    env.scenario_config = config
    env.reset(seed=config.seed)

    assert env.state is not None

    for source, target in config.dependencies:
        assert f"{source}->{target}" in env.state.dependency_states


def test_generated_symptom_changes_runtime_component_state():
    generator = ScenarioGenerator()
    config = generator.generate(
        seed=9903,
        scenario_id="dynamic-test-003",
        scenario_family="configuration",
    )

    env = BranchingDecisionRecoveryEnv()
    env.scenario_config = config
    env.reset(seed=config.seed)

    assert env.state is not None

    symptom_component = {
        "latency_spike": "processing",
        "validation_drift": "validation",
        "dependency_mismatch": "dependency",
        "resource_pressure": "state_store",
    }[config.symptom_profile]

    assert env.state.component_states[symptom_component] == "degraded"


def test_public_scenarios_keep_canonical_behavior():
    env = BranchingDecisionRecoveryEnv(
        scenario_id="decision-v1-public-001",
        seed=3001,
    )

    env.reset()

    assert env.state is not None
    assert env.state.component_states["source"] == "healthy"
    assert env.state.dependency_states["primary_dependency"] == "unknown"
