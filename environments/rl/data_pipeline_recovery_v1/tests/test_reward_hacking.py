from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv
from data_pipeline_recovery_v1.actions import ActionKind


def test_repeated_pipeline_inspection_has_no_reward():
    env = DataPipelineRecoveryEnv()
    env.reset()

    first = env.step(ActionKind.INSPECT_PIPELINE)
    second = env.step(ActionKind.INSPECT_PIPELINE)
    third = env.step(ActionKind.INSPECT_PIPELINE)

    assert first.reward == 1.0
    assert second.reward == 0.0
    assert third.reward == 0.0


def test_repeated_pipeline_inspection_does_not_advance_logical_stage():
    env = DataPipelineRecoveryEnv()
    env.reset()

    env.step(ActionKind.INSPECT_PIPELINE)
    stage_after_first = env.state.logical_stage

    env.step(ActionKind.INSPECT_PIPELINE)
    env.step(ActionKind.INSPECT_PIPELINE)

    assert env.state.logical_stage == stage_after_first


def test_repeated_component_inspection_has_no_reward():
    env = DataPipelineRecoveryEnv()
    env.reset()

    first = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )
    second = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )
    third = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )

    assert first.reward == 0.75
    assert second.reward == 0.0
    assert third.reward == 0.0


def test_different_component_inspections_are_rewarded_once():
    env = DataPipelineRecoveryEnv()
    env.reset()

    first = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )
    second = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="cleaning",
    )
    repeated = env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )

    assert first.reward == 0.75
    assert second.reward == 0.75
    assert repeated.reward == 0.0


def test_repeated_component_inspection_does_not_advance_logical_stage():
    env = DataPipelineRecoveryEnv()
    env.reset()

    env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )
    stage_after_first = env.state.logical_stage

    env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )
    env.step(
        ActionKind.INSPECT_COMPONENT,
        component="schema_validation",
    )

    assert env.state.logical_stage == stage_after_first


def test_twenty_repeated_pipeline_inspections_cannot_farm_reward():
    env = DataPipelineRecoveryEnv()
    env.reset()

    total_reward = 0.0

    for _ in range(20):
        total_reward += env.step(ActionKind.INSPECT_PIPELINE).reward

    assert total_reward == 1.0
