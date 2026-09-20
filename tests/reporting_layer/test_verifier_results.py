import pytest

from agentforge.reporting.verifier_results import VerifierResults


def test_verifier_results():
    result = VerifierResults(10, 8, 2)
    assert result.pass_rate == 0.8


def test_invalid_verifier_counts():
    with pytest.raises(ValueError):
        VerifierResults(10, 9, 0)
