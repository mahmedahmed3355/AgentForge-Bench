from agentforge.reporting.trajectory_analysis import TrajectoryAnalysis


def test_trajectory_analysis():
    analysis = TrajectoryAnalysis([1.0, 2.0, 3.0])
    assert analysis.total_reward == 6.0
    assert analysis.steps == 3
