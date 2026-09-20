import pytest

from agentforge.runtime.registry import TaskRegistry


def test_register_and_get():
    registry = TaskRegistry()

    def factory():
        return "environment"

    registration = registry.register("task-1", factory)

    assert registration.task_id == "task-1"
    assert registry.get("task-1").factory is factory
    assert registry.contains("task-1")


def test_duplicate_registration_rejected():
    registry = TaskRegistry()
    registry.register("task-1", lambda: None)

    with pytest.raises(ValueError):
        registry.register("task-1", lambda: None)


def test_task_ids_are_deterministic():
    registry = TaskRegistry()
    registry.register("task-b", lambda: None)
    registry.register("task-a", lambda: None)

    assert registry.task_ids() == ("task-a", "task-b")
