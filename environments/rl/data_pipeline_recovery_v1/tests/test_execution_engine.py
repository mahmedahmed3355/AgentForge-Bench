from data_pipeline_recovery_v1.actions import ActionKind
from data_pipeline_recovery_v1.decision_graph import IncidentKind
from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv


def run(env, action, **kwargs):
    return env.step(action, **kwargs)


def test_schema_branch_is_stateful_and_long_horizon():
    env = DataPipelineRecoveryEnv(
        episode_id="test-schema",
        incident=IncidentKind.SCHEMA,
    )

    env.reset()

    run(env, ActionKind.INSPECT_PIPELINE)
    run(env, ActionKind.INSPECT_COMPONENT, component="schema_validation")
    run(env, ActionKind.ANALYZE_FAILURE)
    run(env, ActionKind.SELECT_BRANCH, branch="schema_recovery")
    run(env, ActionKind.SELECT_STRATEGY, strategy="repair_upstream")
    run(env, ActionKind.MODIFY_COMPONENT)
    run(env, ActionKind.CHECK_SCHEMA)
    run(env, ActionKind.RUN_PIPELINE)
    run(env, ActionKind.CHECK_OUTPUT)

    assert env.state.root_cause_identified
    assert env.state.selected_strategy == "repair_upstream"
    assert env.state.pipeline_run_completed
    assert env.state.schema_valid
    assert not env.state.terminal


def test_transformation_branch_causes_downstream_failure_and_requires_recovery():
    env = DataPipelineRecoveryEnv(
        episode_id="test-transform",
        incident=IncidentKind.TRANSFORMATION,
    )

    env.reset()

    run(env, ActionKind.INSPECT_PIPELINE)
    run(env, ActionKind.ANALYZE_FAILURE)
    run(env, ActionKind.SELECT_BRANCH, branch="transformation_recovery")
    run(env, ActionKind.SELECT_STRATEGY, strategy="patch_transform")
    run(env, ActionKind.MODIFY_COMPONENT)

    assert env.state.downstream_inconsistency
    assert env.state.replanning_required

    result = run(env, ActionKind.RUN_PIPELINE)

    assert result.info["result"] == "downstream_failure"
    assert env.state.recovery_required
    assert not env.state.terminal

    run(env, ActionKind.REPLAN)
    run(env, ActionKind.RECOVER)

    assert env.state.recovery_completed
    assert not env.state.recovery_required
    assert not env.state.downstream_inconsistency


def test_final_verify_cannot_succeed_from_local_repair_alone():
    env = DataPipelineRecoveryEnv(
        episode_id="test-no-shortcut",
        incident=IncidentKind.SCHEMA,
    )

    env.reset()

    run(env, ActionKind.INSPECT_PIPELINE)
    run(env, ActionKind.ANALYZE_FAILURE)
    run(env, ActionKind.SELECT_BRANCH, branch="schema_recovery")
    run(env, ActionKind.SELECT_STRATEGY, strategy="repair_upstream")
    run(env, ActionKind.MODIFY_COMPONENT)

    result = run(env, ActionKind.FINAL_VERIFY)

    assert result.terminated is False
    assert env.state.terminal is False
    assert not result.info["global_success"]
    assert not env.state.success
