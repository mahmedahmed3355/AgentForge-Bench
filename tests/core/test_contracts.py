from agentforge import (
    Action,
    EpisodeTrace,
    Observation,
    OracleResult,
    RewardResult,
    Scenario,
    StepResult,
    TaskData,
    VerificationResult,
)


def test_task_data_contract():
    task = TaskData(
        idx=0,
        name="test-task",
        description="Contract test",
        prompt="Perform the task",
    )

    assert task.idx == 0
    assert task.name == "test-task"


def test_action_and_observation_contracts():
    action = Action(
        name="inspect",
        parameters={"target": "system"},
    )

    observation = Observation(
        step=0,
        state={"status": "ready"},
        available_actions=("inspect",),
    )

    assert action.name == "inspect"
    assert observation.step == 0
    assert "inspect" in observation.available_actions


def test_step_result_contract():
    observation = Observation(
        step=1,
        state={"status": "running"},
    )

    result = StepResult(
        observation=observation,
        reward=1.5,
        terminated=False,
        truncated=False,
    )

    assert result.reward == 1.5
    assert not result.terminated
    assert not result.truncated


def test_reward_and_verification_contracts():
    reward = RewardResult(
        value=2.0,
        components={"progress": 1.0, "recovery": 1.0},
    )

    verification = VerificationResult(
        passed=True,
        score=1.0,
        metrics={"correctness": 1.0},
    )

    oracle = OracleResult(
        passed=True,
        score=1.0,
        steps=10,
    )

    assert reward.value == 2.0
    assert verification.passed
    assert oracle.passed


def test_scenario_contract():
    scenario = Scenario(
        scenario_id="train-001",
        split="train",
        seed=42,
        parameters={"difficulty": "baseline"},
    )

    assert scenario.scenario_id == "train-001"
    assert scenario.split == "train"
    assert scenario.seed == 42


def test_episode_trace():
    trace = EpisodeTrace(
        task_name="test-task",
        scenario_id="train-001",
        seed=42,
    )

    trace.add(
        step=0,
        event_type="reset",
        data={"status": "ready"},
    )

    trace.add(
        step=1,
        event_type="action",
        data={"name": "inspect"},
    )

    assert len(trace.events) == 2
    assert trace.events[0].event_type == "reset"
    assert trace.events[1].event_type == "action"
