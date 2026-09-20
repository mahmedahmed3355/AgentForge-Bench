from agentforge.reporting.summary import SummaryReport


def test_summary():
    report = SummaryReport(10, 7, 25.0)
    assert report.task_success_rate == 0.7
    assert report.as_dict()["total_reward"] == 25.0
