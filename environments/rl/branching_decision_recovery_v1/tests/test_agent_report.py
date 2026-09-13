from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv
from branching_decision_recovery_v1.report import AgentReport


def test_agent_report_from_state():
    env = BranchingDecisionRecoveryEnv(
        scenario_id="decision-v1-public-001",
        seed=3001,
    )

    env.reset(seed=3001)

    env.step("inspect_system")
    env.step("inspect_component", component="processing")
    env.step("inspect_dependency", dependency="primary_dependency")
    env.step("inspect_history")

    env.step(
        "form_hypothesis",
        hypothesis="downstream_dependency_risk",
        confidence=0.75,
    )

    env.step("select_branch", branch="branch_a")
    env.step("select_strategy", strategy="strategy_a")
    env.step("apply_action")

    for _ in range(8):
        env.step("run")
        if env.state.terminal:
            break

    if env.state.failure_detected:
        env.step("recover", strategy="rollback")
        env.step("replan")
        env.step("select_branch", branch="branch_b")
        env.step("select_strategy", strategy="strategy_b")
        env.step("apply_action")

        for _ in range(4):
            env.step("run")

        env.step("validate")
        env.step("validate")
        env.step("final_verify")

    report = AgentReport.from_state(env.state, task_id="decision-v1-public-001")

    assert report.task_id == "decision-v1-public-001"
    assert report.information_queries >= 4
    assert report.branch_decisions >= 1
    assert report.planning_depth > 0
    assert report.total_cost >= 0
    assert report.final_reward >= 0
    assert report.terminal is True
    assert report.success is True


def test_agent_report_to_dict():
    report = AgentReport(
        task_id="test",
        status="success",
        planning_depth=12,
        information_queries=4,
        branch_decisions=2,
        wrong_decisions=1,
        delayed_failures=1,
        recoveries=1,
        replans=1,
        redundant_actions=0,
        invalid_actions=0,
        total_cost=20,
        final_reward=15.5,
        terminal=True,
        success=True,
    )

    data = report.to_dict()

    assert data["task_id"] == "test"
    assert data["planning_depth"] == 12
    assert data["information_queries"] == 4
    assert data["final_reward"] == 15.5
    assert data["terminal"] is True
    assert data["success"] is True
