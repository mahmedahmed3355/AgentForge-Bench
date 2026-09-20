from agentforge.reporting.generalization import GeneralizationReport


def test_generalization():
    report = GeneralizationReport(8, 10, 6, 10)
    assert report.seen_rate == 0.8
    assert report.unseen_rate == 0.6
