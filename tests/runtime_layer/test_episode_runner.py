from agentforge.runtime.episode_runner import EpisodeRunner


class DemoEnvironment:
    def __init__(self):
        self.current = 0

    def reset(self):
        self.current = 0
        return "state-0", {"reset": True}

    def step(self, action):
        self.current += 1

        terminated = self.current >= 2

        return (
            f"state-{self.current}",
            1.0,
            terminated,
            False,
            {"action": action},
        )


def test_episode_runner_executes_until_terminal():
    runner = EpisodeRunner(DemoEnvironment())

    result = runner.run(["a", "b", "c"])

    assert result.step_count == 2
    assert result.total_reward == 2.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.trajectory == (
        "state-0",
        "state-1",
        "state-2",
    )
