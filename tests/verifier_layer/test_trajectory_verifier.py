from agentforge.verifier.trajectory import verify_trajectory


def test_non_empty_trajectory_is_valid():
    result = verify_trajectory([1, 2, 3])
    assert result.valid is True
    assert result.length == 3


def test_empty_trajectory_is_rejected():
    result = verify_trajectory([])
    assert result.valid is False


def test_none_step_is_rejected():
    result = verify_trajectory([1, None, 3])
    assert result.valid is False
