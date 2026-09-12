from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IncidentKind(str, Enum):
    SCHEMA = "schema"
    TRANSFORMATION = "transformation"
    DATA_QUALITY = "data_quality"
    RESOURCE = "resource"


class RecoveryStrategy(str, Enum):
    REPAIR_UPSTREAM = "repair_upstream"
    REPAIR_CONSUMER = "repair_consumer"

    PATCH_TRANSFORM = "patch_transform"
    ROLLBACK_RECOMPUTE = "rollback_recompute"

    FILTER_INVALID = "filter_invalid"
    RECONSTRUCT_INVALID = "reconstruct_invalid"

    OPTIMIZE_EXECUTION = "optimize_execution"
    RETRY_WITH_RECOVERY = "retry_with_recovery"


class RecoveryMode(str, Enum):
    DIRECT = "direct"
    REPLAN = "replan"


@dataclass(frozen=True)
class StrategyProfile:
    strategy: RecoveryStrategy
    mode: RecoveryMode
    requires_replan: bool
    causes_downstream_validation: bool
    minimum_logical_stages: int


@dataclass(frozen=True)
class Branch:
    name: str
    incident: IncidentKind
    minimum_logical_stages: int
    strategies: tuple[RecoveryStrategy, ...]
    profiles: tuple[StrategyProfile, ...]


BRANCHES = (
    Branch(
        name="schema_recovery",
        incident=IncidentKind.SCHEMA,
        minimum_logical_stages=14,
        strategies=(
            RecoveryStrategy.REPAIR_UPSTREAM,
            RecoveryStrategy.REPAIR_CONSUMER,
        ),
        profiles=(
            StrategyProfile(
                RecoveryStrategy.REPAIR_UPSTREAM,
                RecoveryMode.DIRECT,
                False,
                True,
                14,
            ),
            StrategyProfile(
                RecoveryStrategy.REPAIR_CONSUMER,
                RecoveryMode.REPLAN,
                True,
                True,
                16,
            ),
        ),
    ),
    Branch(
        name="transformation_recovery",
        incident=IncidentKind.TRANSFORMATION,
        minimum_logical_stages=16,
        strategies=(
            RecoveryStrategy.PATCH_TRANSFORM,
            RecoveryStrategy.ROLLBACK_RECOMPUTE,
        ),
        profiles=(
            StrategyProfile(
                RecoveryStrategy.PATCH_TRANSFORM,
                RecoveryMode.REPLAN,
                True,
                True,
                17,
            ),
            StrategyProfile(
                RecoveryStrategy.ROLLBACK_RECOMPUTE,
                RecoveryMode.DIRECT,
                False,
                True,
                16,
            ),
        ),
    ),
    Branch(
        name="data_quality_recovery",
        incident=IncidentKind.DATA_QUALITY,
        minimum_logical_stages=17,
        strategies=(
            RecoveryStrategy.FILTER_INVALID,
            RecoveryStrategy.RECONSTRUCT_INVALID,
        ),
        profiles=(
            StrategyProfile(
                RecoveryStrategy.FILTER_INVALID,
                RecoveryMode.REPLAN,
                True,
                True,
                18,
            ),
            StrategyProfile(
                RecoveryStrategy.RECONSTRUCT_INVALID,
                RecoveryMode.DIRECT,
                False,
                True,
                17,
            ),
        ),
    ),
    Branch(
        name="resource_recovery",
        incident=IncidentKind.RESOURCE,
        minimum_logical_stages=15,
        strategies=(
            RecoveryStrategy.OPTIMIZE_EXECUTION,
            RecoveryStrategy.RETRY_WITH_RECOVERY,
        ),
        profiles=(
            StrategyProfile(
                RecoveryStrategy.OPTIMIZE_EXECUTION,
                RecoveryMode.REPLAN,
                True,
                True,
                17,
            ),
            StrategyProfile(
                RecoveryStrategy.RETRY_WITH_RECOVERY,
                RecoveryMode.DIRECT,
                False,
                True,
                15,
            ),
        ),
    ),
)


def get_branch(incident: IncidentKind) -> Branch:
    for branch in BRANCHES:
        if branch.incident == incident:
            return branch
    raise ValueError(f"Unsupported incident: {incident}")


def get_strategy_profile(
    incident: IncidentKind,
    strategy: RecoveryStrategy,
) -> StrategyProfile:
    branch = get_branch(incident)

    for profile in branch.profiles:
        if profile.strategy == strategy:
            return profile

    raise ValueError(
        f"Strategy {strategy.value!r} is not valid for incident "
        f"{incident.value!r}"
    )
