import pytest

from agentforge.reporting.episode_summary import EpisodeSummary


def test_episode_summary():
    item = EpisodeSummary("ep-1", 3.5, 7, True, False)
    assert item.episode_id == "ep-1"
    assert item.steps == 7


def test_negative_steps_rejected():
    with pytest.raises(ValueError):
        EpisodeSummary("ep-1", 0.0, -1, False, False)
