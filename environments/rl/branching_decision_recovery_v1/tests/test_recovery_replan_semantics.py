from branching_decision_recovery_v1.actions import ActionKind
from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv


def build_failed_episode():
    env = BranchingDecisionRecoveryEnv()
    env.reset(seed=3001)

    env.step(ActionKind.INSPECT_SYSTEM)
    env.step(ActionKind.INSPECT_COMPONENT, component="processing")
    env.step(ActionKind.INSPECT_DEPENDENCY, dependency="processing")
    env.step(ActionKind.INSPECT_HISTORY)

    env.step(
        ActionKind.FORM_HYPOTHESIS,
        hypothesis="downstream_dependency_risk",
        confidence=0.75,
    )
    env.step(ActionKind.SELECT_BRANCH, branch="branch_a")
    env.step(ActionKind.SELECT_STRATEGY, strategy="strategy_a")
    env.step(ActionKind.APPLY_ACTION)

    for _ in range(6):
        env.step(ActionKind.RUN)

    assert env.state is not None
    assert env.state.failure_detected is True
    assert env.state.recovery_required is True
    assert env.state.replanning_required is True
    assert len(env.state.failure_history) >= 1

    return env


def test_recovery_clears_active_failure_but_preserves_history():
    env = build_failed_episode()

    env.step(ActionKind.RECOVER, strategy="rollback", cost=8.0)

    assert env.state is not None
    assert env.state.failure_detected is False
    assert env.state.failure_source is None
    assert env.state.downstream_effect is None
    assert env.state.recovery_required is False
    assert env.state.recovery_completed is True
    assert len(env.state.failure_history) >= 1
    assert env.state.validation_status == "not_validated"


def test_replan_is_not_terminal_success():
    env = build_failed_episode()

    env.step(ActionKind.RECOVER, strategy="rollback", cost=8.0)
    env.step(ActionKind.REPLAN)

    assert env.state is not None
    assert env.state.failure_detected is False
    assert env.state.replanning_required is False
    assert env.state.replan_completed is True
    assert env.state.terminal is False
    assert env.state.success is False
    assert env.state.validation_status == "not_validated"


def test_final_verify_cannot_succeed_without_global_validation():
    env = build_failed_episode()

    env.step(ActionKind.RECOVER, strategy="rollback", cost=8.0)
    env.step(ActionKind.REPLAN)

    _, reward, terminal, _ = env.step(ActionKind.FINAL_VERIFY)

    assert reward == 0.0
    assert terminal is False
    assert env.state is not None
    assert env.state.success is False


def test_recovery_replan_requires_fresh_execution_and_validation():
    env = build_failed_episode()

    env.step(ActionKind.RECOVER, strategy="rollback", cost=8.0)
    env.step(ActionKind.REPLAN)

    env.step(ActionKind.RUN)

    assert env.state is not None
    assert env.state.terminal is False
    assert env.state.validation_status == "not_validated"

    env.step(ActionKind.VALIDATE)

    assert env.state is not None
    assert env.state.validation_status == "local_validated"
    assert env.state.terminal is False
