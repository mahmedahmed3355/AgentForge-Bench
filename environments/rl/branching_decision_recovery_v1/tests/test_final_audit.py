from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv
from branching_decision_recovery_v1.report import AgentReport
from branching_decision_recovery_v1.oracle import BranchingDecisionOracle
from branching_decision_recovery_v1.verifier import BranchingDecisionVerifier


FORBIDDEN_MARKERS = {
    "oracle_answer",
    "hidden_answer",
    "reference_solution",
    "reference_answer",
    "ground_truth",
    "hidden_solution",
    "secret_solution",
    "expected_optimal",
}


def step(env, action, **kwargs):
    observation, reward, terminal, info = env.step(action, **kwargs)
    return float(reward), bool(terminal), info


def run_reference_recovery_episode():
    env = BranchingDecisionRecoveryEnv(
        scenario_id="decision-v1-public-001",
        seed=3001,
    )

    env.reset(seed=3001)

    rewards = []

    reward, _, _ = step(env, "inspect_system")
    rewards.append(reward)

    reward, _, _ = step(
        env,
        "inspect_component",
        component="processing",
    )
    rewards.append(reward)

    reward, _, _ = step(
        env,
        "inspect_dependency",
        dependency="primary_dependency",
    )
    rewards.append(reward)

    reward, _, _ = step(env, "inspect_history")
    rewards.append(reward)

    reward, _, _ = step(
        env,
        "form_hypothesis",
        hypothesis="downstream_dependency_risk",
        confidence=0.75,
    )
    rewards.append(reward)

    reward, _, _ = step(
        env,
        "select_branch",
        branch="branch_a",
    )
    rewards.append(reward)

    reward, _, _ = step(
        env,
        "select_strategy",
        strategy="strategy_a",
    )
    rewards.append(reward)

    reward, _, _ = step(env, "apply_action")
    rewards.append(reward)

    for _ in range(32):
        reward, terminal, _ = step(env, "run")
        rewards.append(reward)

        if env.state.failure_detected or terminal:
            break

    if env.state.failure_detected:
        reward, _, _ = step(
            env,
            "recover",
            strategy="rollback",
        )
        rewards.append(reward)

        reward, _, _ = step(env, "replan")
        rewards.append(reward)

        reward, _, _ = step(
            env,
            "select_branch",
            branch="branch_b",
        )
        rewards.append(reward)

        reward, _, _ = step(
            env,
            "select_strategy",
            strategy="strategy_b",
        )
        rewards.append(reward)

        reward, _, _ = step(env, "apply_action")
        rewards.append(reward)

        for _ in range(4):
            reward, terminal, _ = step(env, "run")
            rewards.append(reward)

            if terminal:
                break

        if not env.state.terminal:
            reward, _, _ = step(env, "validate")
            rewards.append(reward)

        if not env.state.terminal:
            reward, _, _ = step(env, "validate")
            rewards.append(reward)

        if not env.state.terminal:
            reward, _, _ = step(env, "final_verify")
            rewards.append(reward)

    return env, rewards


def test_final_audit_reference_episode():
    env, _ = run_reference_recovery_episode()

    oracle = BranchingDecisionOracle()
    verifier = BranchingDecisionVerifier()

    oracle_result = oracle.evaluate(env.state)
    verifier_result = verifier.verify(env.state)

    assert env.state.terminal is True
    assert env.state.success is True
    assert oracle_result.solved is True

    verifier_dict = verifier_result.__dict__
    assert any(
        value is True
        for key, value in verifier_dict.items()
        if key in {"verified", "passed", "success", "solved"}
    )


def test_final_audit_report_matches_trajectory():
    env, rewards = run_reference_recovery_episode()

    report = AgentReport.from_state(
        env.state,
        task_id="decision-v1-public-001",
        final_reward=float(sum(rewards)),
    )

    assert report.task_id == "decision-v1-public-001"
    assert report.terminal is True
    assert report.success is True

    assert report.planning_depth == env.state.logical_stage
    assert report.final_reward == float(sum(rewards))

    assert report.information_queries == len(
        env.state.observed_information
    )

    assert report.recoveries == len(
        env.state.recovery_history
    )

    assert report.invalid_actions >= 0
    assert report.redundant_actions >= 0
    assert report.total_cost >= 0

    report_dict = report.to_dict()

    required_fields = {
        "task_id",
        "status",
        "planning_depth",
        "information_queries",
        "branch_decisions",
        "wrong_decisions",
        "delayed_failures",
        "recoveries",
        "replans",
        "redundant_actions",
        "invalid_actions",
        "total_cost",
        "final_reward",
        "terminal",
        "success",
    }

    assert set(report_dict) == required_fields


def test_final_audit_report_is_frozen():
    env, _ = run_reference_recovery_episode()

    report = AgentReport.from_state(
        env.state,
        task_id="decision-v1-public-001",
    )

    try:
        report.success = False
    except Exception:
        pass
    else:
        raise AssertionError(
            "AgentReport must be immutable"
        )


def test_final_audit_no_hidden_truth_in_agent_package():
    from pathlib import Path

    package_root = Path(
        "environments/rl/branching_decision_recovery_v1/"
        "branching_decision_recovery_v1"
    )

    for path in package_root.glob("*.py"):
        if path.name in {
            "observations.py",
        }:
            continue

        text = path.read_text().lower()

        for marker in FORBIDDEN_MARKERS:
            assert marker not in text, (
                f"Hidden-truth marker {marker!r} found in {path}"
            )


def test_final_audit_report_contains_no_hidden_truth():
    report_path = (
        "environments/rl/branching_decision_recovery_v1/"
        "branching_decision_recovery_v1/report.py"
    )

    text = open(report_path).read().lower()

    for marker in FORBIDDEN_MARKERS:
        assert marker not in text


def test_final_audit_report_serialization():
    env, _ = run_reference_recovery_episode()

    report = AgentReport.from_state(
        env.state,
        task_id="decision-v1-public-001",
    )

    data = report.to_dict()

    assert isinstance(data, dict)
    assert isinstance(data["task_id"], str)
    assert isinstance(data["status"], str)
    assert isinstance(data["planning_depth"], int)
    assert isinstance(data["information_queries"], int)
    assert isinstance(data["branch_decisions"], int)
    assert isinstance(data["wrong_decisions"], int)
    assert isinstance(data["delayed_failures"], int)
    assert isinstance(data["recoveries"], int)
    assert isinstance(data["replans"], int)
    assert isinstance(data["redundant_actions"], int)
    assert isinstance(data["invalid_actions"], int)
    assert isinstance(data["total_cost"], int)
    assert isinstance(data["final_reward"], float)
    assert isinstance(data["terminal"], bool)
    assert isinstance(data["success"], bool)
