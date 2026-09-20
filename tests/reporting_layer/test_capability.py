from agentforge.reporting.capability import CapabilityReport


def test_capability():
    report = CapabilityReport("cuda", 10, 8)
    assert report.success_rate == 0.8


def test_capability_from_mapping():
    report = CapabilityReport.from_mapping(
        "backend",
        {"attempts": 4, "successes": 3},
    )
    assert report.success_rate == 0.75
