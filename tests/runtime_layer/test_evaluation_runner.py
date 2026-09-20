from agentforge.runtime.evaluation_runner import EvaluationRunner
from agentforge.runtime.factory import EnvironmentFactory
from agentforge.runtime.registry import TaskRegistry


class DemoEnvironment:
    def reset(self):
        return "state", {}

    def step(self, action):
        return "next", float(action), True, False, {}

    def close(self):
        pass


def test_evaluation_runner_aggregates_episodes():
    registry = TaskRegistry()
    registry.register(
        "demo",
        lambda: DemoEnvironment(),
    )

    factory = EnvironmentFactory(registry)
    runner = EvaluationRunner(factory)

    result = runner.run(
        "demo",
        [
            [1],
            [2],
            [3],
        ],
    )

    assert result.episode_count == 3
    assert result.total_reward == 6.0
    assert result.mean_reward == 2.0
