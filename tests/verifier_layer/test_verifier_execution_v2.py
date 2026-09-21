from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from agentforge.verifier import (
    VerificationResult,
    VerifierContext,
    VerifierExecutor,
    VerifierRegistry,
)


@dataclass(frozen=True)
class FakeVerifier:
    verifier_id: str = "fake-verifier"
    calls: list[VerifierContext] = field(default_factory=list, compare=False)

    def verify(self, context: VerifierContext) -> VerificationResult:
        self.calls.append(context)

        return VerificationResult(
            verified=True,
            verifier_id=self.verifier_id,
            task_id=context.task_spec.identity.task_id,
            episode=context.episode_result,
            trajectory=context.trajectory,
            reward={"total": context.episode_result.total_reward},
            success={"success": True},
            diagnostics={"failures": ()},
            metadata=context.metadata,
        )


def make_task(task_id: str = "TEST-TASK"):
    return SimpleNamespace(
        identity=SimpleNamespace(task_id=task_id),
        verifier=SimpleNamespace(
            verifier_id="fake-verifier",
            implementation=None,
            configuration={},
        ),
    )


def make_episode():
    return SimpleNamespace(
        total_reward=3.5,
        step_count=4,
        terminated=True,
        truncated=False,
    )


def test_registry_resolves_verifier_spec():
    verifier = FakeVerifier()
    registry = VerifierRegistry()
    registry.register("fake-verifier", verifier)

    resolved = registry.resolve("fake-verifier")

    assert resolved is verifier


def test_executor_resolves_task_verifier_spec_and_executes():
    verifier = FakeVerifier()
    registry = VerifierRegistry()
    registry.register("fake-verifier", verifier)

    task = make_task()
    scenario = {"scenario_id": "scenario-1"}
    episode = make_episode()
    trajectory = ("step-0", "step-1")

    result = VerifierExecutor(registry=registry).execute(
        task_spec=task,
        scenario=scenario,
        episode_result=episode,
        trajectory=trajectory,
        metadata={"seed": 42},
    )

    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert result.task_id == "TEST-TASK"
    assert result.verifier_id == "fake-verifier"
    assert result.episode is episode
    assert result.trajectory == trajectory
    assert result.metadata["seed"] == 42

    assert len(verifier.calls) == 1
    context = verifier.calls[0]

    assert context.task_spec is task
    assert context.scenario is scenario
    assert context.episode_result is episode
    assert context.trajectory == trajectory


def test_executor_rejects_unknown_verifier():
    registry = VerifierRegistry()

    with pytest.raises(KeyError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario={},
            episode_result=make_episode(),
            trajectory=(),
        )


def test_executor_requires_task_verifier():
    registry = VerifierRegistry()
    registry.register("fake-verifier", FakeVerifier())

    task = SimpleNamespace(
        identity=SimpleNamespace(task_id="NO-VERIFIER"),
    )

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=task,
            scenario={},
            episode_result=make_episode(),
            trajectory=(),
        )


def test_executor_rejects_mismatched_result_task_id():
    class BadTaskVerifier:
        verifier_id = "fake-verifier"

        def verify(self, context):
            return VerificationResult(
                verified=True,
                verifier_id="fake-verifier",
                task_id="WRONG-TASK",
                episode=context.episode_result,
                trajectory=context.trajectory,
                reward={},
                success={},
                diagnostics={},
            )

    registry = VerifierRegistry()
    registry.register("fake-verifier", BadTaskVerifier())

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario={},
            episode_result=make_episode(),
            trajectory=(),
        )


def test_executor_rejects_mismatched_result_verifier_id():
    class BadVerifier:
        verifier_id = "fake-verifier"

        def verify(self, context):
            return VerificationResult(
                verified=True,
                verifier_id="wrong-verifier",
                task_id="TEST-TASK",
                episode=context.episode_result,
                trajectory=context.trajectory,
                reward={},
                success={},
                diagnostics={},
            )

    registry = VerifierRegistry()
    registry.register("fake-verifier", BadVerifier())

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario={},
            episode_result=make_episode(),
            trajectory=(),
        )
