from agentforge.reporting.oracle_comparison import OracleComparison


def test_oracle_comparison():
    comparison = OracleComparison(10.0, 7.5)
    assert comparison.reward_gap == 2.5
    assert comparison.exact_match is False
