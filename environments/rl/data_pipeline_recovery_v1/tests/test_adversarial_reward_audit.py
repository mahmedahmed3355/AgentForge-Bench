from data_pipeline_recovery_v1.actions import ActionKind
from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv


def make_env():
    return DataPipelineRecoveryEnv(
        episode_id="reward-audit",
    )


def test_inspect_pipeline_x20_cannot_farm_reward():
    env = make_env()
    env.reset()

    rewards = []
    stages = []

    for _ in range(20):
        result = env.step(ActionKind.INSPECT_PIPELINE)
        rewards.append(result.reward)
        stages.append(env.state.logical_stage)

    assert rewards[0] > 0
    assert all(reward == 0 for reward in rewards[1:])
    assert stages[0] == stages[-1]


def test_run_pipeline_x20_cannot_farm_reward():
    env = make_env()
    env.reset()

    rewards = []

    for _ in range(20):
        result = env.step(ActionKind.RUN_PIPELINE)
        rewards.append(result.reward)

    assert all(reward <= 0 for reward in rewards)
    assert sum(rewards) <= 0
    assert env.state.success is False


def test_check_output_x20_cannot_farm_reward():
    env = make_env()
    env.reset()

    rewards = []

    for _ in range(20):
        result = env.step(ActionKind.CHECK_OUTPUT)
        rewards.append(result.reward)

    assert rewards[0] <= 0
    assert all(reward <= 0 for reward in rewards)
    assert sum(rewards) <= 0
    assert env.state.success is False


def test_recover_without_failure_cannot_create_success():
    env = make_env()
    env.reset()

    result = env.step(ActionKind.RECOVER)

    assert result.reward <= 0
    assert env.state.recovery_completed is False
    assert env.state.success is False
    assert env.state.terminal is False


def test_final_verify_early_cannot_create_success():
    env = make_env()
    env.reset()

    result = env.step(ActionKind.FINAL_VERIFY)

    assert result.reward < 10.0
    assert env.state.success is False
    assert env.state.terminal is False


def test_partial_completion_cannot_receive_final_success():
    env = make_env()

    env.reset()

    env.step(ActionKind.INSPECT_PIPELINE)
    env.step(
        ActionKind.SELECT_BRANCH,
        branch="schema_recovery",
    )

    result = env.step(ActionKind.FINAL_VERIFY)

    assert result.reward < 10.0
    assert env.state.success is False
    assert env.state.terminal is False


def test_repeated_inspections_do_not_accumulate_meaningful_reward():
    env = make_env()
    env.reset()

    total_reward = 0.0

    for _ in range(20):
        result = env.step(ActionKind.INSPECT_PIPELINE)
        total_reward += result.reward

    assert total_reward <= 1.0
    assert env.state.success is False


def test_mixed_reward_farming_sequence_cannot_succeed():
    env = make_env()
    env.reset()

    actions = (
        ActionKind.INSPECT_PIPELINE,
        ActionKind.INSPECT_PIPELINE,
        ActionKind.INSPECT_PIPELINE,
        ActionKind.RUN_PIPELINE,
        ActionKind.RUN_PIPELINE,
        ActionKind.CHECK_OUTPUT,
        ActionKind.CHECK_OUTPUT,
        ActionKind.RECOVER,
        ActionKind.RECOVER,
        ActionKind.FINAL_VERIFY,
    )

    rewards = []

    for action in actions:
        result = env.step(action)
        rewards.append(result.reward)

    assert env.state.success is False
    assert env.state.terminal is False
    assert max(rewards) < 10.0
