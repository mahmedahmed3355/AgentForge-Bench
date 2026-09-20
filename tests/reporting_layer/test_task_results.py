from agentforge.reporting.task_results import TaskResultReport


def test_task_result_serialization():
    report = TaskResultReport(
        task_id="task-001",
        success=True,
        reward=1.0,
        episodes=3,
        details={"mode": "oracle"},
    )
    data = report.as_dict()
    assert data["task_id"] == "task-001"
    assert data["details"]["mode"] == "oracle"
