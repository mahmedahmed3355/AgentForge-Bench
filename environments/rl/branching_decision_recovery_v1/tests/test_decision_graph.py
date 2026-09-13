from branching_decision_recovery_v1.decision_graph import (
    BRANCH_PROFILES,
    STRATEGY_PROFILES,
)


def test_multiple_plausible_branches_exist():
    assert len(BRANCH_PROFILES) >= 3


def test_multiple_strategies_exist():
    assert len(STRATEGY_PROFILES) >= 3


def test_delayed_consequences_exist():
    assert any(profile.delayed_consequence for profile in BRANCH_PROFILES)


def test_branches_have_different_costs():
    assert len({profile.cost for profile in BRANCH_PROFILES}) > 1
