from __future__ import annotations

import pytest
from agentforge.contracts.task import (
    ActionSpec,
    CompatibilitySpec,
    EnvironmentBinding,
    ObservationSpec,
    OracleSpec,
    RewardSpec,
    ScenarioSpec,
    SuccessSpec,
    TaskIdentity,
    TaskSpec,
    TerminationSpec,
    VerifierSpec,
)
from agentforge.runtime.task_registry import (
    RegisteredTask,
    TaskRegistry,
)


def make_spec(task_id: str = "TEST-001") -> TaskSpec:
    return TaskSpec(
        identity=TaskIdentity(
            task_id=task_id,
            version="1.0.0",
            domain="test",
            split="train",
        ),
        environment=EnvironmentBinding(
            environment_id="test-env",
            version="1.0.0",
        ),
        scenario=ScenarioSpec(
            scenario_id="test-scenario",
        ),
        actions=ActionSpec(
            vocabulary=("noop",),
        ),
        observations=ObservationSpec(
            schema={"type": "integer"},
        ),
        reward=RewardSpec(
            specification={"type": "scalar"},
        ),
        termination=TerminationSpec(
            max_episode_steps=10,
        ),
        success=SuccessSpec(
            criteria=("success",),
        ),
        oracle=OracleSpec(oracle_id="test-oracle", 
 
            configuration={"enabled": True},
        ),
        verifier=VerifierSpec(verifier_id="test-verifier", 
            configuration={"enabled": True},
        ),
        compatibility=CompatibilitySpec(
            contract_version="1.0",
        ),
    )


def environment_factory():
    return object()


def scenario_factory():
    return {"scenario": True}


def reward_factory():
    return {"reward": True}


def test_registered_task_contains_runtime_bindings():
    task = RegisteredTask(
        spec=make_spec(),
        environment_factory=environment_factory,
        scenario_factory=scenario_factory,
        reward_factory=reward_factory,
        oracle=object(),
        verifier=object(),
    )
    assert task.task_id == "TEST-001"
    assert task.version == "1.0.0"
    assert task.environment_factory is environment_factory
    assert task.scenario_factory is scenario_factory
    assert task.reward_factory is reward_factory
    assert task.has_evaluator
    assert task.has_oracle
    assert task.has_verifier


def test_registered_task_runtime_factories():
    task = RegisteredTask(
        spec=make_spec(),
        environment_factory=environment_factory,
        scenario_factory=scenario_factory,
        reward_factory=reward_factory,
    )
    assert task.make_environment() is not None
    assert task.make_scenario() == {"scenario": True}
    assert task.make_reward_engine() == {"reward": True}


def test_missing_optional_factory_fails_at_use():
    task = RegisteredTask(
        spec=make_spec(),
        environment_factory=environment_factory,
    )
    with pytest.raises(RuntimeError, match="scenario_factory"):
        task.make_scenario()
    with pytest.raises(RuntimeError, match="reward_factory"):
        task.make_reward_engine()


def test_registry_register_get_and_contains():
    registry = TaskRegistry()
    task = RegisteredTask(
        spec=make_spec("TEST-002"),
        environment_factory=environment_factory,
    )
    registry.register(task)
    assert len(registry) == 1
    assert registry.contains("TEST-002")
    assert "TEST-002" in registry
    assert registry.get("TEST-002") is task


def test_registry_rejects_duplicate_registration():
    registry = TaskRegistry()
    task = RegisteredTask(
        spec=make_spec("TEST-003"),
        environment_factory=environment_factory,
    )
    registry.register(task)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(task)


def test_registry_order_is_deterministic():
    registry = TaskRegistry()
    registry.register(
        RegisteredTask(
            spec=make_spec("TEST-B"),
            environment_factory=environment_factory,
        )
    )
    registry.register(
        RegisteredTask(
            spec=make_spec("TEST-A"),
            environment_factory=environment_factory,
        )
    )
    assert registry.ids() == ("TEST-A", "TEST-B")
    assert tuple(t.task_id for t in registry.all()) == (
        "TEST-A",
        "TEST-B",
    )


def test_registry_explicit_replace_and_unregister():
    registry = TaskRegistry()
    first = RegisteredTask(
        spec=make_spec("TEST-004"),
        environment_factory=environment_factory,
    )
    second = RegisteredTask(
        spec=make_spec("TEST-004"),
        environment_factory=lambda: "replacement",
    )
    registry.register(first)
    registry.replace(second)
    assert registry.get("TEST-004") is second
    registry.unregister("TEST-004")
    assert not registry.contains("TEST-004")
    with pytest.raises(KeyError, match="not registered"):
        registry.get("TEST-004")
