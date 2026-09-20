import pytest

from agentforge.reporting.coverage import CoverageReport


def test_coverage_ratio():
    assert CoverageReport(10, 8).coverage_ratio == 0.8


def test_invalid_coverage():
    with pytest.raises(ValueError):
        CoverageReport(5, 6).coverage_ratio
