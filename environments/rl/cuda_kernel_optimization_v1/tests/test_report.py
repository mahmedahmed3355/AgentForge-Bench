from environments.rl.cuda_kernel_optimization_v1.cuda_kernel_optimization_v1.report import (
    AgentEpisodeReport,
    EpisodeReporter,
)


def test_episode_report_records_execution():
    reporter = EpisodeReporter(
        task_id="task-001",
        scenario_id="train-001",
        seed=42,
        episode_id="episode-test",
    )

    reporter.start()

    reporter.record_step(
        action="inspect",
        reward=0.2,
        observation={
            "logical_stage": 2,
            "success": False,
        },
        terminal=False,
        success=False,
        info={
            "decision": "tiling",
            "failure": None,
        },
    )

    reporter.add_tokens(
        input_tokens=100,
        output_tokens=50,
    )

    reporter.record_step(
        action="recover",
        reward=0.3,
        observation={
            "logical_stage": 5,
            "success": False,
        },
        terminal=False,
        success=False,
        info={
            "failure": "correctness",
        },
    )

    report = reporter.finalize(success=True)

    assert report.episode_id == "episode-test"
    assert report.task_id == "task-001"
    assert report.scenario_id == "train-001"
    assert report.seed == 42

    assert report.action_count == 2
    assert report.total_reward == 0.5
    assert report.logical_stages_reached == 5
    assert report.recovery_count == 1
    assert report.failure_count == 1

    assert report.token_usage.input_tokens == 100
    assert report.token_usage.output_tokens == 50
    assert report.token_usage.total_tokens == 150

    assert report.success is True
    assert report.status == "success"

    assert report.decisions[0]["decision"] == "tiling"
    assert report.failures[0]["type"] == "correctness"

    payload = report.to_dict()

    assert payload["task_id"] == "task-001"
    assert payload["token_usage"]["total_tokens"] == 150
    assert len(payload["trajectory"]) == 2


def test_report_json_is_serializable():
    report = AgentEpisodeReport(
        episode_id="episode-json",
        task_id="task-001",
        scenario_id=None,
        seed=7,
    )

    report.finalize(success=False)

    text = report.to_json()

    assert '"task_id": "task-001"' in text
    assert '"success": false' in text
