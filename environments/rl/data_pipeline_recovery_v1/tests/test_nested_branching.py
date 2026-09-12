from data_pipeline_recovery_v1.actions import ActionKind
from data_pipeline_recovery_v1.decision_graph import (
    IncidentKind,
    RecoveryMode,
    RecoveryStrategy,
    get_branch,
    get_strategy_profile,
)
from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv


def prepare_env(incident: IncidentKind) -> DataPipelineRecoveryEnv:
    env = DataPipelineRecoveryEnv(
        episode_id=f"test-{incident.value}",
        incident=incident,
    )

    env.reset()

    env.step(ActionKind.INSPECT_PIPELINE)
    env.step(
        ActionKind.INSPECT_COMPONENT,
        component="input",
    )
    env.step(ActionKind.ANALYZE_FAILURE)

    return env


def test_each_incident_has_two_real_strategies():
    for incident in IncidentKind:
        branch = get_branch(incident)

        assert len(branch.strategies) == 2

        for strategy in branch.strategies:
            profile = get_strategy_profile(
                incident,
                strategy,
            )
            assert profile.strategy == strategy
            assert profile.minimum_logical_stages >= 14


def test_schema_has_direct_and_replan_paths():
    direct = get_strategy_profile(
        IncidentKind.SCHEMA,
        RecoveryStrategy.REPAIR_UPSTREAM,
    )

    nested = get_strategy_profile(
        IncidentKind.SCHEMA,
        RecoveryStrategy.REPAIR_CONSUMER,
    )

    assert direct.mode == RecoveryMode.DIRECT
    assert direct.requires_replan is False

    assert nested.mode == RecoveryMode.REPLAN
    assert nested.requires_replan is True


def test_transformation_patch_creates_nested_replanning():
    env = prepare_env(IncidentKind.TRANSFORMATION)

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="transformation_recovery",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="patch_transform",
    )

    result = env.step(
        ActionKind.MODIFY_COMPONENT,
        component="transformation",
    )

    assert result.reward == 1.25
    assert env.state.transformation_valid is True
    assert env.state.downstream_inconsistency is True
    assert env.state.replanning_required is True


def test_transformation_rollback_has_different_path():
    env = prepare_env(IncidentKind.TRANSFORMATION)

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="transformation_recovery",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="rollback_recompute",
    )

    env.step(
        ActionKind.MODIFY_COMPONENT,
        component="transformation",
    )

    assert env.state.recovery_mode == RecoveryMode.DIRECT
    assert env.state.replanning_required is False
    assert env.state.downstream_inconsistency is True


def test_wrong_branch_is_rejected():
    env = prepare_env(IncidentKind.DATA_QUALITY)

    result = env.step(
        ActionKind.SELECT_BRANCH,
        branch="schema_recovery",
    )

    assert result.reward < 0
    assert env.state.selected_branch is None


def test_wrong_strategy_is_rejected():
    env = prepare_env(IncidentKind.RESOURCE)

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="resource_recovery",
    )

    result = env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="patch_transform",
    )

    assert result.reward < 0
    assert env.state.selected_strategy is None


def test_local_repair_cannot_bypass_global_validation():
    env = prepare_env(IncidentKind.DATA_QUALITY)

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="data_quality_recovery",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="filter_invalid",
    )

    env.step(
        ActionKind.MODIFY_COMPONENT,
        component="cleaning",
    )

    result = env.step(
        ActionKind.FINAL_VERIFY,
    )

    assert result.reward == -5.0
    assert env.state.success is False
    assert env.state.terminal is False
    assert result.terminated is False


def test_nested_replan_then_recovery_changes_state():
    env = prepare_env(IncidentKind.DATA_QUALITY)

    env.step(
        ActionKind.SELECT_BRANCH,
        branch="data_quality_recovery",
    )

    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="filter_invalid",
    )

    env.step(
        ActionKind.MODIFY_COMPONENT,
        component="cleaning",
    )

    assert env.state.replanning_required is True

    env.step(ActionKind.REPLAN)

    assert env.state.replan_completed is True
    assert env.state.replanning_required is False

    env.step(ActionKind.RUN_PIPELINE)

    assert env.state.recovery_required is True
    assert env.state.failure_detected is True

    env.step(ActionKind.RECOVER)

    assert env.state.recovery_completed is True
    assert env.state.recovery_required is False
    assert env.state.downstream_inconsistency is False
