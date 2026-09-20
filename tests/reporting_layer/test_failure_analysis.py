import pytest

from agentforge.reporting.failure_analysis import FailureAnalysis


def test_failure_analysis():
    analysis = FailureAnalysis()
    analysis.add("timeout")
    analysis.add("invalid action")
    assert analysis.count == 2
    assert analysis.as_dict()["failures"] == [
        "timeout",
        "invalid action",
    ]


def test_empty_reason_rejected():
    with pytest.raises(ValueError):
        FailureAnalysis().add("")
