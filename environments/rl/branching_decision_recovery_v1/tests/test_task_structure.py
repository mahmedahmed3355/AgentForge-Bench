from pathlib import Path

from branching_decision_recovery_v1.actions import ActionKind
from branching_decision_recovery_v1.decision_graph import (
    BRANCH_PROFILES,
    STRATEGY_PROFILES,
    DecisionBranch,
    DecisionStrategy,
)


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        "README.md",
        "pyproject.toml",
        "data/scenarios.json",
        "docs/DESIGN.md",
        "docs/TASK_CONTRACT.md",
        "docs/RL_REQUIREMENTS.md",
        "branching_decision_recovery_v1/__init__.py",
        "branching_decision_recovery_v1/actions.py",
        "branching_decision_recovery_v1/decision_graph.py",
        "branching_decision_recovery_v1/state.py",
        "branching_decision_recovery_v1/observations.py",
        "branching_decision_recovery_v1/inspect_adapter.py",
        "branching_decision_recovery_v1/reward.py",
        "branching_decision_recovery_v1/trajectory.py",
        "branching_decision_recovery_v1/environment.py",
        "branching_decision_recovery_v1/gymnasium_env.py",
        "branching_decision_recovery_v1/oracle.py",
        "branching_decision_recovery_v1/verifier.py",
        "branching_decision_recovery_v1/report.py",
        "branching_decision_recovery_v1/prime_state.py",
        "branching_decision_recovery_v1/prime_toolset.py",
        "branching_decision_recovery_v1/taskset.py",
        "branching_decision_recovery_v1/prime_environment.py",
        "evaluator/hidden_scenarios.py",
        "evaluator/harness.py",
        "evaluator/runner.py",
    ]

    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing, f"Missing required files: {missing}"


def test_action_space_contains_core_mechanisms():
    required = {
        ActionKind.INSPECT_SYSTEM,
        ActionKind.INSPECT_COMPONENT,
        ActionKind.INSPECT_DEPENDENCY,
        ActionKind.INSPECT_HISTORY,
        ActionKind.PROBE_STATE,
        ActionKind.QUERY_VALIDATION,
        ActionKind.FORM_HYPOTHESIS,
        ActionKind.SELECT_BRANCH,
        ActionKind.SELECT_STRATEGY,
        ActionKind.APPLY_ACTION,
        ActionKind.MODIFY_CONFIGURATION,
        ActionKind.RUN,
        ActionKind.VALIDATE,
        ActionKind.RECOVER,
        ActionKind.REPLAN,
        ActionKind.FINAL_VERIFY,
    }

    assert required.issubset(set(ActionKind))


def test_multiple_plausible_branches_exist():
    assert len(BRANCH_PROFILES) >= 3

    branches = {profile.branch for profile in BRANCH_PROFILES}

    assert DecisionBranch.BRANCH_A in branches
    assert DecisionBranch.BRANCH_B in branches
    assert DecisionBranch.BRANCH_C in branches


def test_multiple_strategies_exist():
    assert len(STRATEGY_PROFILES) >= 3

    strategies = {profile.strategy for profile in STRATEGY_PROFILES}

    assert DecisionStrategy.STRATEGY_A in strategies
    assert DecisionStrategy.STRATEGY_B in strategies
    assert DecisionStrategy.STRATEGY_C in strategies


def test_delayed_consequences_are_present():
    delayed = [
        profile
        for profile in BRANCH_PROFILES
        if profile.delayed_consequence
    ]

    assert delayed, "At least one branch must have a delayed consequence"


def test_cost_and_risk_are_not_uniform():
    branch_costs = {profile.cost for profile in BRANCH_PROFILES}
    branch_risks = {profile.risk for profile in BRANCH_PROFILES}

    assert len(branch_costs) > 1
    assert len(branch_risks) > 1
