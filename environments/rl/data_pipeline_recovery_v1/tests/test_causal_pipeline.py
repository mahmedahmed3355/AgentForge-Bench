from data_pipeline_recovery_v1.actions import ActionKind
from data_pipeline_recovery_v1.causal import (
    CausalPipeline,
    PipelineComponent,
)
from data_pipeline_recovery_v1.decision_graph import IncidentKind
from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv


def test_pipeline_has_explicit_dependency_chain():
    pipeline = CausalPipeline()

    assert pipeline.dependencies(
        PipelineComponent.SCHEMA
    ) == (PipelineComponent.INPUT,)

    assert pipeline.dependencies(
        PipelineComponent.CLEANING
    ) == (PipelineComponent.SCHEMA,)

    assert pipeline.dependencies(
        PipelineComponent.TRANSFORMATION
    ) == (PipelineComponent.CLEANING,)

    assert pipeline.dependencies(
        PipelineComponent.AGGREGATION
    ) == (PipelineComponent.TRANSFORMATION,)


def test_invalid_upstream_component_blocks_downstream_components():
    pipeline = CausalPipeline()

    pipeline.set_valid(
        PipelineComponent.SCHEMA,
        True,
    )

    run = pipeline.run()

    assert run.completed
    assert "cleaning" in run.invalid_components
    assert "transformation" in run.blocked_components
    assert "aggregation" in run.blocked_components


def test_local_transformation_repair_does_not_make_pipeline_global_valid():
    env = DataPipelineRecoveryEnv(
        episode_id="causal-transform",
        incident=IncidentKind.TRANSFORMATION,
    )

    env.reset()

    env.step(ActionKind.INSPECT_PIPELINE)
    env.step(ActionKind.ANALYZE_FAILURE)
    env.step(
        ActionKind.SELECT_BRANCH,
        branch="transformation_recovery",
    )
    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="patch_transform",
    )
    env.step(ActionKind.MODIFY_COMPONENT)

    assert env.state.transformation_valid
    assert env.state.downstream_inconsistency

    result = env.step(ActionKind.RUN_PIPELINE)

    assert result.info["result"] == "downstream_failure"
    assert result.info["invalid_components"]
    assert env.state.recovery_required


def test_causal_state_is_reflected_in_public_state():
    env = DataPipelineRecoveryEnv(
        episode_id="causal-schema",
        incident=IncidentKind.SCHEMA,
    )

    env.reset()

    env.step(ActionKind.INSPECT_PIPELINE)
    env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )
    env.step(ActionKind.ANALYZE_FAILURE)
    env.step(
        ActionKind.SELECT_BRANCH,
        branch="schema_recovery",
    )
    env.step(
        ActionKind.SELECT_STRATEGY,
        strategy="repair_upstream",
    )
    env.step(ActionKind.MODIFY_COMPONENT)

    assert env.state.component_validity[
        "schema_validation"
    ] is True

    assert "cleaning" in env.state.blocked_components or (
        env.state.component_validity["cleaning"] is False
    )
