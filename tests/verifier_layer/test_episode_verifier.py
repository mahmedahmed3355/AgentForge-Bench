from agentforge.verifier.episode import verify_episode


def test_valid_terminated_episode():
    result = verify_episode(
        step_count=3,
        terminated=True,
        truncated=False,
    )
    assert result.valid is True


def test_valid_truncated_episode():
    result = verify_episode(
        step_count=3,
        terminated=False,
        truncated=True,
    )
    assert result.valid is True


def test_negative_step_count_rejected():
    result = verify_episode(
        step_count=-1,
        terminated=True,
        truncated=False,
    )
    assert result.valid is False


def test_double_terminal_state_rejected():
    result = verify_episode(
        step_count=1,
        terminated=True,
        truncated=True,
    )
    assert result.valid is False
