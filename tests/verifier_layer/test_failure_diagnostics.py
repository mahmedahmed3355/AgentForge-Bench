from agentforge.verifier.diagnostics import diagnose_failures


def test_no_failures():
    result = diagnose_failures(
        [
            ("episode", True),
            ("reward", True),
            ("trajectory", True),
        ]
    )
    assert result.failed is False
    assert result.failures == ()


def test_failures_are_recorded():
    result = diagnose_failures(
        [
            ("episode", True),
            ("reward", False),
            ("trajectory", False),
        ]
    )
    assert result.failed is True
    assert result.failures == ("reward", "trajectory")
