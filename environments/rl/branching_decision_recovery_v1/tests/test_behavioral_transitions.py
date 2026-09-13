from branching_decision_recovery_v1.actions import ActionKind
from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv


def make_env():
    env = BranchingDecisionRecoveryEnv(
        scenario_id="decision-v1-public-001",
        seed=3001,
    )
    env.reset()
    return env


def test_information_gathering_changes_observation_but_not_hidden_truth():
    env = make_env()

    before = env.state.logical_stage

    observation, reward, terminal, info = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="processing",
    )

    assert reward > 0.0
    assert terminal is False
    assert info["inspection"]["new_information"] is True
    assert env.state.logical_stage > before

    assert "component:processing" in observation.observed_information
    assert "incident_family" not in observation.observed_information
    assert "incident_family" not in observation.to_dict()


def test_redundant_information_gathering_has_no_progress_reward():
    env = make_env()

    _, first_reward, _, first_info = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="processing",
    )

    stage_after_first = env.state.logical_stage
    cost_after_first = env.state.decision_cost

    _, second_reward, _, second_info = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="processing",
    )

    assert first_reward > 0.0
    assert second_reward == 0.0
    assert first_info["inspection"]["new_information"] is True
    assert second_info["inspection"]["new_information"] is False
    assert env.state.logical_stage == stage_after_first
    assert env.state.decision_cost == cost_after_first


def test_hypothesis_precedes_branch_decision():
    env = make_env()

    _, hypothesis_reward, _, _ = env.step(
        ActionKind.FORM_HYPOTHESIS,
        hypothesis="dependency_failure",
        confidence=0.8,
    )

    stage_after_hypothesis = env.state.logical_stage

    _, branch_reward, _, _ = env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_a",
    )

    assert hypothesis_reward > 0.0
    assert branch_reward > 0.0
    assert env.state.hypothesis == "dependency_failure"
    assert env.state.hypothesis_confidence == 0.8
    assert env.state.selected_branch == "branch_a"
    assert env.state.logical_stage > stage_after_hypothesis


def test_costly_action_requires_a_branch():
    env = make_env()

    before_budget = env.state.remaining_budget

    _, reward, terminal, info = env.step(
        ActionKind.APPLY_ACTION,
        cost=5.0,
    )

    assert reward < 0.0
    assert terminal is False
    assert info["valid"] is True
    assert env.state.selected_branch is None
    assert env.state.remaining_budget == before_budget - 5.0


def test_delayed_consequence_is_not_immediate():
    env = make_env()

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_a",
    )

    _, _, _, _ = env.step(
        ActionKind.APPLY_ACTION,
        cost=1.0,
    )

    assert env.state.failure_detected is False
    assert env.state.recovery_required is False
    assert env.state.replanning_required is False

    env.step(ActionKind.RUN)

    assert env.state.failure_detected is False

    env.step(ActionKind.RUN)
    env.step(ActionKind.RUN)

    assert env.state.failure_detected is False


def test_delayed_consequence_eventually_creates_failure():
    env = make_env()

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_a",
    )

    env.step(
        ActionKind.APPLY_ACTION,
        cost=1.0,
    )

    for _ in range(3):
        env.step(ActionKind.RUN)

    assert env.state.failure_detected is False
    assert env.state.recovery_required is False
    assert env.state.replanning_required is False

    for _ in range(3):
        env.step(ActionKind.RUN)

    assert env.state.failure_detected is True
    assert env.state.recovery_required is True
    assert env.state.replanning_required is True
    assert env.state.failure_source == "downstream_dependency"
    assert env.state.downstream_effect is not None


def test_failure_preserves_historical_trajectory():
    env = make_env()

    env.step(ActionKind.INSPECT_SYSTEM)
    env.step(ActionKind.INSPECT_COMPONENT, component="processing")
    env.step(ActionKind.SELECT_BRANCH, branch="branch_a")
    env.step(ActionKind.APPLY_ACTION, cost=1.0)

    actions_before_failure = list(env.state.action_history)

    for _ in range(6):
        env.step(ActionKind.RUN)

    assert env.state.failure_detected is True

    assert env.state.action_history[: len(actions_before_failure)] == (
        actions_before_failure
    )

    assert len(env.state.failure_history) >= 1
    assert env.state.step == len(env.state.action_history)


def test_recovery_does_not_reset_state_or_history():
    env = make_env()

    env.step(ActionKind.INSPECT_SYSTEM)
    env.step(ActionKind.SELECT_BRANCH, branch="branch_a")
    env.step(ActionKind.APPLY_ACTION, cost=1.0)

    for _ in range(6):
        env.step(ActionKind.RUN)

    assert env.state.recovery_required is True

    history_before_recovery = list(env.state.action_history)
    failure_history_before_recovery = list(env.state.failure_history)
    stage_before_recovery = env.state.logical_stage

    _, reward, terminal, _ = env.step(
        ActionKind.RECOVER,
        strategy="reconfigure",
        cost=1.0,
    )

    assert reward > 0.0
    assert terminal is False
    assert env.state.recovery_completed is True
    assert env.state.recovery_required is False

    assert env.state.action_history[: len(history_before_recovery)] == (
        history_before_recovery
    )
    assert env.state.failure_history == failure_history_before_recovery
    assert env.state.logical_stage > stage_before_recovery


def test_replan_is_required_after_delayed_failure():
    env = make_env()

    env.step(ActionKind.SELECT_BRANCH, branch="branch_a")
    env.step(ActionKind.APPLY_ACTION, cost=1.0)

    for _ in range(6):
        env.step(ActionKind.RUN)

    assert env.state.replanning_required is True

    _, reward, terminal, _ = env.step(ActionKind.REPLAN)

    assert reward > 0.0
    assert terminal is False
    assert env.state.replan_completed is True
    assert env.state.replanning_required is False


def test_global_validation_is_distinct_from_local_validation():
    env = make_env()

    env.step(ActionKind.SELECT_BRANCH, branch="branch_b")
    env.step(ActionKind.SELECT_STRATEGY, strategy="strategy_b")

    _, local_reward, terminal, _ = env.step(ActionKind.VALIDATE)

    assert local_reward > 0.0
    assert terminal is False
    assert env.state.validation_status == "local_validated"

    _, final_reward, terminal, _ = env.step(ActionKind.FINAL_VERIFY)

    assert final_reward == 0.0
    assert terminal is False
    assert env.state.success is False


def test_success_requires_branch_strategy_and_global_validation():
    env = make_env()

    env.step(ActionKind.SELECT_BRANCH, branch="branch_b")
    env.step(ActionKind.SELECT_STRATEGY, strategy="strategy_b")

    env.state.validation_status = "globally_validated"

    _, reward, terminal, _ = env.step(ActionKind.FINAL_VERIFY)

    assert reward == 10.0
    assert terminal is True
    assert env.state.success is True
    assert env.state.system_status == "success"


def test_failure_blocks_premature_final_success():
    env = make_env()

    env.step(ActionKind.SELECT_BRANCH, branch="branch_a")
    env.step(ActionKind.APPLY_ACTION, cost=1.0)

    for _ in range(6):
        env.step(ActionKind.RUN)

    env.state.validation_status = "globally_validated"

    _, reward, terminal, _ = env.step(ActionKind.FINAL_VERIFY)

    assert reward == -5.0
    assert terminal is True
    assert env.state.success is False
    assert env.state.system_status == "unrecoverable_failure"
