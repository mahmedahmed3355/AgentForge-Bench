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
    TaskContract,
    TaskEnvironment,
    TaskIdentity,
    TaskSpec,
    TerminationSpec,
    VerifierSpec,
)


def make_task_spec() -> TaskSpec:
    return TaskSpec(
        identity=TaskIdentity(
            task_id="example-task",
            version="1.0.0",
            domain="backend",
            split="train",
        ),
        environment=EnvironmentBinding(
            environment_id="agentforge/example",
            environment_version="1",
            max_episode_steps=20,
        ),
        scenario=ScenarioSpec(
            scenario_id="example-scenario",
            generator="example.generator",
            parameters={"seeded": True},
        ),
        actions=ActionSpec(
            vocabulary=("inspect", "repair", "verify"),
            schema={"type": "string"},
        ),
        observations=ObservationSpec(
            schema={"type": "object"},
            description="Structured task observation.",
        ),
        reward=RewardSpec(
            kind="sparse",
            specification={"success": 1.0, "failure": 0.0},
        ),
        termination=TerminationSpec(
            termination_conditions=("success", "failure"),
            truncation_conditions=("timeout",),
            max_episode_steps=20,
        ),
        success=SuccessSpec(
            criteria=("system_state_is_correct",),
            success_reward=1.0,
        ),
        oracle=OracleSpec(
            oracle_id="example-oracle",
            implementation="example.oracle:solve",
        ),
        verifier=VerifierSpec(
            verifier_id="example-verifier",
            implementation="example.verifier:verify",
        ),
        compatibility=CompatibilitySpec(
            contract_version="1.0",
            gymnasium_version="1.3",
            framework_version="1.0",
        ),
        objective="Repair the target system.",
        difficulty=0.5,
        horizon=20,
        capabilities=("python", "debugging"),
        metadata={"authoring": "benchmark"},
    )


def test_task_spec_contains_canonical_architecture():
    spec = make_task_spec()

    assert spec.identity.task_id == "example-task"
    assert spec.environment.environment_id == "agentforge/example"
    assert spec.scenario.scenario_id == "example-scenario"
    assert spec.actions.vocabulary == (
        "inspect",
        "repair",
        "verify",
    )
    assert spec.observations.schema["type"] == "object"
    assert spec.reward.kind == "sparse"
    assert spec.termination.max_episode_steps == 20
    assert spec.success.criteria == (
        "system_state_is_correct",
    )
    assert spec.oracle.oracle_id == "example-oracle"
    assert spec.verifier.verifier_id == "example-verifier"
    assert spec.compatibility.contract_version == "1.0"


def test_task_spec_identity_compatibility_accessors():
    spec = make_task_spec()

    assert spec.task_id == "example-task"
    assert spec.version == "1.0.0"
    assert spec.domain == "backend"
    assert spec.split == "train"
    assert spec.max_episode_steps == 20


def test_task_spec_rejects_negative_difficulty():
    with pytest.raises(
        ValueError,
        match="difficulty must be non-negative",
    ):
        spec = make_task_spec()
        TaskSpec(
            identity=spec.identity,
            environment=spec.environment,
            scenario=spec.scenario,
            actions=spec.actions,
            observations=spec.observations,
            reward=spec.reward,
            termination=spec.termination,
            success=spec.success,
            oracle=spec.oracle,
            verifier=spec.verifier,
            compatibility=spec.compatibility,
            difficulty=-1,
        )


def test_task_spec_rejects_invalid_horizon():
    with pytest.raises(
        ValueError,
        match="horizon must be positive",
    ):
        spec = make_task_spec()
        TaskSpec(
            identity=spec.identity,
            environment=spec.environment,
            scenario=spec.scenario,
            actions=spec.actions,
            observations=spec.observations,
            reward=spec.reward,
            termination=spec.termination,
            success=spec.success,
            oracle=spec.oracle,
            verifier=spec.verifier,
            compatibility=spec.compatibility,
            horizon=0,
        )


def test_task_contract_remains_stable():
    contract = TaskContract(
        task_id="task-001",
        version="1.0.0",
        max_episode_steps=10,
    )

    assert contract.task_id == "task-001"
    assert contract.version == "1.0.0"
    assert contract.max_episode_steps == 10


def test_task_environment_remains_protocol():
    assert getattr(TaskEnvironment, "__protocol_attrs__", None) is not None


def test_all_canonical_components_are_frozen():
    spec = make_task_spec()

    with pytest.raises(Exception):
        spec.reward = RewardSpec(kind="dense")
