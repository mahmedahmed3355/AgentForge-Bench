from branching_decision_recovery_v1.actions import ActionKind
from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv


def test_reset_creates_partially_observable_state():
    env = BranchingDecisionRecoveryEnv()

    observation = env.reset()

    assert observation.step == 0
    assert observation.logical_stage == 0
    assert observation.system_status == "healthy"
    assert observation.available_actions
    assert "branch_a" not in observation.observed_information


def test_information_gathering_reveals_information_without_revealing_truth():
    env = BranchingDecisionRecoveryEnv()
    observation = env.reset()

    observation, reward, terminal, info = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="processing",
    )

    assert reward > 0
    assert not terminal
    assert info["inspection"]["new_information"] is True
    assert "component:processing" in observation.observed_information
    assert "incident_family" not in observation.observed_information


def test_repeated_information_query_does_not_farm_reward():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    _, first_reward, _, _ = env.step(
        ActionKind.INSPECT_SYSTEM,
    )

    _, second_reward, _, _ = env.step(
        ActionKind.INSPECT_SYSTEM,
    )

    assert first_reward > 0
    assert second_reward == 0.0


def test_branch_decision_creates_progress():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    before = env.state.logical_stage

    observation, reward, _, _ = env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_a",
    )

    assert reward > 0
    assert env.state.selected_branch == "branch_a"
    assert observation.logical_stage > before


def test_delayed_failure_requires_recovery_and_replan():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    env.step(ActionKind.SELECT_BRANCH, branch="branch_a")
    env.step(ActionKind.APPLY_ACTION, cost=1.0)

    for _ in range(6):
        env.step(ActionKind.RUN)

    assert env.state.failure_detected
    assert env.state.recovery_required
    assert env.state.replanning_required


def test_recovery_preserves_episode_history():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    env.step(ActionKind.SELECT_BRANCH, branch="branch_a")
    env.step(ActionKind.APPLY_ACTION, cost=1.0)

    for _ in range(6):
        env.step(ActionKind.RUN)

    previous_actions = list(env.state.action_history)
    previous_failures = list(env.state.failure_history)

    env.step(
        ActionKind.RECOVER,
        strategy="reconfigure",
        cost=1.0,
    )

    assert env.state.action_history[: len(previous_actions)] == previous_actions
    assert env.state.failure_history == previous_failures
    assert env.state.recovery_completed


def test_final_success_requires_global_validation():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    env.step(ActionKind.SELECT_BRANCH, branch="branch_b")
    env.step(ActionKind.SELECT_STRATEGY, strategy="strategy_b")

    _, reward, terminal, _ = env.step(ActionKind.FINAL_VERIFY)

    assert reward == 0.0
    assert not terminal

    env.state.validation_status = "globally_validated"

    _, reward, terminal, _ = env.step(ActionKind.FINAL_VERIFY)

    assert reward == 10.0
    assert terminal
    assert env.state.success
