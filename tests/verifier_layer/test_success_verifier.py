from agentforge.verifier.success import verify_success


def test_all_checks_pass():
    result = verify_success(
        episode_valid=True,
        reward_valid=True,
        trajectory_valid=True,
    )
    assert result.success is True


def test_episode_failure_blocks_success():
    result = verify_success(
        episode_valid=False,
        reward_valid=True,
        trajectory_valid=True,
    )
    assert result.success is False


def test_reward_failure_blocks_success():
    result = verify_success(
        episode_valid=True,
        reward_valid=False,
        trajectory_valid=True,
    )
    assert result.success is False


def test_trajectory_failure_blocks_success():
    result = verify_success(
        episode_valid=True,
        reward_valid=True,
        trajectory_valid=False,
    )
    assert result.success is False
