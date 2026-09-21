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
class FakeEpisode:
    total_reward: float = 4.0
    step_count: int = 3
    terminated: bool = True
    truncated: bool = False
    observation: object = None
    trajectory: tuple = ()


@dataclass
class FakeVerifier:
    verifier_id: str = "integration-verifier"
    contexts: list[VerifierContext] = field(default_factory=list)

    def verify(self, context: VerifierContext) -> VerificationResult:
        self.contexts.append(context)

        return VerificationResult(
            verified=True,
            verifier_id=self.verifier_id,
            task_id=context.task_spec.identity.task_id,
            episode=context.episode_result,
            trajectory=context.trajectory,
            reward={"verifier_observed_reward": 4.0},
            success={"verifier_success": True},
            diagnostics={"verifier_failures": ()},
            metadata={"verifier": "fake"},
        )


def make_task(task_id="V3-TASK"):
    return SimpleNamespace(
        identity=SimpleNamespace(task_id=task_id),
        verifier=SimpleNamespace(
            verifier_id="integration-verifier",
            implementation=None,
            configuration={},
        ),
    )


def make_valid_episode():
    return FakeEpisode(
        total_reward=4.0,
        step_count=3,
        terminated=True,
        truncated=False,
    )


def test_v3_runs_all_verification_components():
    verifier = FakeVerifier()
    registry = VerifierRegistry()
    registry.register("integration-verifier", verifier)

    episode = make_valid_episode()
    trajectory = (
        {"step": 0},
        {"step": 1},
        {"step": 2},
    )

    result = VerifierExecutor(registry=registry).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "scenario-v3"},
        episode_result=episode,
        trajectory=trajectory,
        metadata={"seed": 123},
    )

    assert isinstance(result, VerificationResult)
    assert result.verified is True

    assert result.reward.valid is True
    assert result.reward.value == 4.0

    assert result.success.success is True

    assert result.diagnostics.failed is False
    assert result.diagnostics.failures == ()

    assert result.metadata["episode_verification"].valid is True
    assert result.metadata["trajectory_verification"].valid is True
    assert result.metadata["reward_verification"].valid is True
    assert result.metadata["success_verification"].success is True

    assert len(verifier.contexts) == 1
    assert verifier.contexts[0].episode_result is episode
    assert verifier.contexts[0].trajectory == trajectory


def test_v3_detects_invalid_episode():
    verifier = FakeVerifier()
    registry = VerifierRegistry()
    registry.register("integration-verifier", verifier)

    episode = FakeEpisode(
        total_reward=4.0,
        step_count=-1,
        terminated=True,
        truncated=False,
    )

    result = VerifierExecutor(registry=registry).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "scenario-v3"},
        episode_result=episode,
        trajectory=(),
    )

    assert result.verified is False
    assert result.success.success is False
    assert result.diagnostics.failed is True
    assert "episode" in result.diagnostics.failures


def test_v3_detects_invalid_reward():
    verifier = FakeVerifier()
    registry = VerifierRegistry()
    registry.register("integration-verifier", verifier)

    episode = FakeEpisode(
        total_reward=float("nan"),
        step_count=3,
        terminated=True,
        truncated=False,
    )

    result = VerifierExecutor(registry=registry).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "scenario-v3"},
        episode_result=episode,
        trajectory=(),
    )

    assert result.verified is False
    assert result.reward.valid is False
    assert result.success.success is False
    assert result.diagnostics.failed is True
    assert "reward" in result.diagnostics.failures


def test_v3_requires_all_runtime_inputs():
    verifier = FakeVerifier()
    registry = VerifierRegistry()
    registry.register("integration-verifier", verifier)

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario={"scenario_id": "scenario-v3"},
            episode_result=None,
            trajectory=(),
        )

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario=None,
            episode_result=make_valid_episode(),
            trajectory=(),
        )


def test_v3_preserves_verifier_context():
    verifier = FakeVerifier()
    registry = VerifierRegistry()
    registry.register("integration-verifier", verifier)

    scenario = {
        "scenario_id": "scenario-42",
        "configuration": {"mode": "hidden"},
    }

    trajectory = (
        {"step": 0, "action": "A"},
        {"step": 1, "action": "B"},
    )

    VerifierExecutor(registry=registry).execute(
        task_spec=make_task("CONTEXT-TASK"),
        scenario=scenario,
        episode_result=make_valid_episode(),
        trajectory=trajectory,
        metadata={"seed": 77},
    )

    context = verifier.contexts[0]

    assert context.task_spec.identity.task_id == "CONTEXT-TASK"
    assert context.scenario is scenario
    assert context.trajectory == trajectory
    assert context.metadata["seed"] == 77
