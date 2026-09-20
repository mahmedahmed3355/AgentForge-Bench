from agentforge.reporting.performance import PerformanceReport


def test_performance():
    report = PerformanceReport(10, 25.0, 7)
    assert report.mean_reward == 2.5
    assert report.success_rate == 0.7
