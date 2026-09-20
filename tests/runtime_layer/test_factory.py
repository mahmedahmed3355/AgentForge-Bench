from agentforge.runtime.factory import EnvironmentFactory
from agentforge.runtime.registry import TaskRegistry


def test_factory_creates_registered_environment():
    registry = TaskRegistry()

    def factory(value=0):
        return {"value": value}

    registry.register("task-1", factory)

    environment_factory = EnvironmentFactory(registry)

    environment = environment_factory.create(
        "task-1",
        value=42,
    )

    assert environment == {"value": 42}
