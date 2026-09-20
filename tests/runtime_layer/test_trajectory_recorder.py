import pytest

from agentforge.runtime.trajectory_recorder import TrajectoryRecorder


def test_records_steps():
    recorder = TrajectoryRecorder()

    step = recorder.record(
        observation="obs",
        action="act",
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"x": 1},
    )

    assert step.reward == 1.0
    assert len(recorder.steps()) == 1


def test_terminal_flags_cannot_both_be_true():
    recorder = TrajectoryRecorder()

    with pytest.raises(ValueError):
        recorder.record(
            observation="obs",
            action="act",
            reward=1.0,
            terminated=True,
            truncated=True,
        )


def test_closed_recorder_rejects_new_steps():
    recorder = TrajectoryRecorder()
    recorder.close()

    with pytest.raises(RuntimeError):
        recorder.record(
            observation="obs",
            action="act",
            reward=1.0,
            terminated=False,
            truncated=False,
        )
