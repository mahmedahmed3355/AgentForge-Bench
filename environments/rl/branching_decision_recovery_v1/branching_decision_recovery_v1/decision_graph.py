from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecisionBranch(str, Enum):
    BRANCH_A = "branch_a"
    BRANCH_B = "branch_b"
    BRANCH_C = "branch_c"


class DecisionStrategy(str, Enum):
    STRATEGY_A = "strategy_a"
    STRATEGY_B = "strategy_b"
    STRATEGY_C = "strategy_c"


class FailureKind(str, Enum):
    LOCAL = "local"
    DEPENDENCY = "dependency"
    DOWNSTREAM = "downstream"
    CASCADE = "cascade"


class RecoveryStrategy(str, Enum):
    ROLLBACK = "rollback"
    PATCH = "patch"
    RECONFIGURE = "reconfigure"
    RESTORE_DEPENDENCY = "restore_dependency"
    RECOMPUTE = "recompute"
    ISOLATE_COMPONENT = "isolate_component"


@dataclass(frozen=True)
class BranchProfile:
    branch: DecisionBranch
    cost: float
    risk: float
    delayed_consequence: bool
    recovery_required: bool
    consequence_delay: int
    failure_kind: FailureKind | None
    downstream_effect: str


@dataclass(frozen=True)
class StrategyProfile:
    strategy: DecisionStrategy
    cost: float
    risk: float
    requires_information: bool
    compatible_branches: tuple[DecisionBranch, ...]


BRANCH_PROFILES = (
    BranchProfile(
        branch=DecisionBranch.BRANCH_A,
        cost=10.0,
        risk=2.0,
        delayed_consequence=True,
        recovery_required=True,
        consequence_delay=6,
        failure_kind=FailureKind.DOWNSTREAM,
        downstream_effect="A locally valid decision fails at a downstream dependency.",
    ),
    BranchProfile(
        branch=DecisionBranch.BRANCH_B,
        cost=15.0,
        risk=1.0,
        delayed_consequence=False,
        recovery_required=False,
        consequence_delay=0,
        failure_kind=None,
        downstream_effect="The decision remains stable but consumes more resources.",
    ),
    BranchProfile(
        branch=DecisionBranch.BRANCH_C,
        cost=6.0,
        risk=3.0,
        delayed_consequence=True,
        recovery_required=True,
        consequence_delay=4,
        failure_kind=FailureKind.CASCADE,
        downstream_effect="A delayed dependency cascade invalidates the decision.",
    ),
)


STRATEGY_PROFILES = (
    StrategyProfile(
        strategy=DecisionStrategy.STRATEGY_A,
        cost=5.0,
        risk=2.0,
        requires_information=True,
        compatible_branches=(
            DecisionBranch.BRANCH_A,
            DecisionBranch.BRANCH_C,
        ),
    ),
    StrategyProfile(
        strategy=DecisionStrategy.STRATEGY_B,
        cost=9.0,
        risk=1.0,
        requires_information=False,
        compatible_branches=(
            DecisionBranch.BRANCH_A,
            DecisionBranch.BRANCH_B,
            DecisionBranch.BRANCH_C,
        ),
    ),
    StrategyProfile(
        strategy=DecisionStrategy.STRATEGY_C,
        cost=3.0,
        risk=4.0,
        requires_information=True,
        compatible_branches=(
            DecisionBranch.BRANCH_B,
            DecisionBranch.BRANCH_C,
        ),
    ),
)


def get_branch_profile(branch: DecisionBranch | str) -> BranchProfile:
    branch = DecisionBranch(branch)

    for profile in BRANCH_PROFILES:
        if profile.branch == branch:
            return profile

    raise ValueError(f"Unknown decision branch: {branch}")


def get_strategy_profile(
    strategy: DecisionStrategy | str,
) -> StrategyProfile:
    strategy = DecisionStrategy(strategy)

    for profile in STRATEGY_PROFILES:
        if profile.strategy == strategy:
            return profile

    raise ValueError(f"Unknown decision strategy: {strategy}")


def is_strategy_compatible(
    branch: DecisionBranch | str,
    strategy: DecisionStrategy | str,
) -> bool:
    branch_profile = get_branch_profile(branch)
    strategy_profile = get_strategy_profile(strategy)

    return branch_profile.branch in strategy_profile.compatible_branches


def validate_graph() -> None:
    branches = {profile.branch for profile in BRANCH_PROFILES}
    strategies = {profile.strategy for profile in STRATEGY_PROFILES}

    expected_branches = set(DecisionBranch)
    expected_strategies = set(DecisionStrategy)

    if branches != expected_branches:
        raise ValueError("Decision graph branches are incomplete.")

    if strategies != expected_strategies:
        raise ValueError("Decision graph strategies are incomplete.")

    for profile in BRANCH_PROFILES:
        if profile.cost < 0:
            raise ValueError("Branch cost cannot be negative.")
        if profile.risk < 0:
            raise ValueError("Branch risk cannot be negative.")
        if profile.delayed_consequence and profile.consequence_delay <= 0:
            raise ValueError(
                "Delayed branches must have a positive consequence delay."
            )

    for profile in STRATEGY_PROFILES:
        if profile.cost < 0:
            raise ValueError("Strategy cost cannot be negative.")
        if profile.risk < 0:
            raise ValueError("Strategy risk cannot be negative.")
        if not profile.compatible_branches:
            raise ValueError(
                "Every strategy must support at least one branch."
            )


validate_graph()
