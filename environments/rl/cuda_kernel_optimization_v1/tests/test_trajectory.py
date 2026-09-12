from cuda_kernel_optimization_v1.trajectory import Trajectory


def test_trajectory_records_transitions():
    trajectory = Trajectory()

    trajectory.append(
        step=1,
        stage_before=0,
        stage_after=2,
        action="inspect",
        reward=0.2,
        success=False,
        terminal=False,
    )

    trajectory.append(
        step=2,
        stage_before=2,
        stage_after=5,
        action="modify_tiling",
        reward=0.5,
        success=False,
        terminal=False,
    )

    assert trajectory.action_count == 2
    assert trajectory.total_reward == 0.7
    assert not trajectory.terminal_success


def test_trajectory_counts_recovery():
    trajectory = Trajectory()

    trajectory.append(
        step=1,
        stage_before=5,
        stage_after=5,
        action="recover",
        reward=0.3,
        success=False,
        terminal=False,
    )

    assert trajectory.recovery_count == 1
