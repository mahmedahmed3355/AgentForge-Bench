from branching_decision_recovery_v1.actions import ActionKind
from branching_decision_recovery_v1.decision_graph import (
    DecisionBranch,
    DecisionStrategy,
    get_branch_profile,
    get_strategy_profile,
    is_strategy_compatible,
)
from branching_decision_recovery_v1.environment import (
    BranchingDecisionRecoveryEnv,
)


def test_branch_profiles_define_distinct_consequences():
    a = get_branch_profile(DecisionBranch.BRANCH_A)
    b = get_branch_profile(DecisionBranch.BRANCH_B)
    c = get_branch_profile(DecisionBranch.BRANCH_C)

    assert a.delayed_consequence is True
    assert b.delayed_consequence is False
    assert c.delayed_consequence is True

    assert a.consequence_delay != c.consequence_delay
    assert a.failure_kind != c.failure_kind
    assert b.failure_kind is None


def test_strategy_compatibility_is_explicit():
    assert is_strategy_compatible(
        DecisionBranch.BRANCH_A,
        DecisionStrategy.STRATEGY_A,
    )

    assert is_strategy_compatible(
        DecisionBranch.BRANCH_B,
        DecisionStrategy.STRATEGY_B,
    )

    assert is_strategy_compatible(
        DecisionBranch.BRANCH_C,
        DecisionStrategy.STRATEGY_C,
    )

    assert not is_strategy_compatible(
        DecisionBranch.BRANCH_B,
        DecisionStrategy.STRATEGY_A,
    )


def test_incompatible_strategy_blocks_costly_action():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_b",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="strategy_a",
    )

    before_budget = env.state.remaining_budget

    _, reward, terminal, info = env.step(
        ActionKind.APPLY_ACTION,
        cost=1.0,
    )

    assert reward < 0.0
    assert terminal is False
    assert env.state.system_status == "invalid_strategy"
    assert env.state.remaining_budget == before_budget
    assert info["valid"] is True


def test_branch_b_has_no_delayed_failure():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_b",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="strategy_b",
    )

    env.step(
        ActionKind.APPLY_ACTION,
        cost=1.0,
    )

    for _ in range(8):
        env.step(ActionKind.RUN)

    assert env.state.failure_detected is False
    assert env.state.recovery_required is False
    assert env.state.replanning_required is False
    assert env.state.system_status == "stable"


def test_branch_a_uses_graph_delay_and_failure_kind():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_a",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="strategy_b",
    )

    env.step(
        ActionKind.APPLY_ACTION,
        cost=1.0,
    )

    assert len(env.state.delayed_events) == 1

    event = env.state.delayed_events[0]

    assert event["type"] == "downstream"
    assert event["source"] == "branch_a"

    for _ in range(5):
        env.step(ActionKind.RUN)

    assert env.state.failure_detected is False

    env.step(ActionKind.RUN)

    assert env.state.failure_detected is True
    assert env.state.failure_source == "downstream_dependency"


def test_branch_c_has_different_delayed_consequence():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    env.step(
        ActionKind.INSPECT_COMPONENT,
        component="processing",
    )

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="branch_c",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="strategy_c",
    )

    env.step(
        ActionKind.APPLY_ACTION,
        cost=1.0,
    )

    event = env.state.delayed_events[0]

    assert event["type"] == "cascade"
    assert event["source"] == "branch_c"

    for _ in range(3):
        env.step(ActionKind.RUN)

    assert env.state.failure_detected is False

    env.step(ActionKind.RUN)

    assert env.state.failure_detected is True
    assert env.state.failure_source == "dependency_cascade"
