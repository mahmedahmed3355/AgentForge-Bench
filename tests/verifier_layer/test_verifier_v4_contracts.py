
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from agentforge.verifier import (
    BaseVerifier,
    FailureDiagnostics,
    VerificationResult,
    VerifierContext,
    VerifierExecutor,
    VerifierRegistry,
)
from agentforge.verifier.episode import EpisodeVerification, verify_episode
from agentforge.verifier.reward import RewardVerification, verify_reward
from agentforge.verifier.success import SuccessVerification, verify_success
from agentforge.verifier.trajectory import (
    TrajectoryVerification,
    verify_trajectory,
)


@dataclass(frozen=True)
class FakeEpisodeResult:
    observation: object = None
    total_reward: float = 5.0
    step_count: int = 3
    terminated: bool = True
    truncated: bool = False
    trajectory: tuple = ()


@dataclass
class RecordingVerifier:
    verifier_id: str = "v4-verifier"
    contexts: list[VerifierContext] = field(default_factory=list)

    def verify(self, context: VerifierContext) -> VerificationResult:
        self.contexts.append(context)

        return VerificationResult(
            verified=True,
            verifier_id=self.verifier_id,
            task_id=context.task_spec.identity.task_id,
            episode=context.episode_result,
            trajectory=context.trajectory,
            reward={"source": "verifier"},
            success={"source": "verifier"},
            diagnostics={"source": "verifier"},
            metadata={"source": "verifier"},
        )


def make_task(
    task_id: str = "V4-TASK",
    verifier_id: str = "v4-verifier",
):
    return SimpleNamespace(
        identity=SimpleNamespace(task_id=task_id),
        verifier=SimpleNamespace(
            verifier_id=verifier_id,
            implementation=None,
            configuration={"mode": "isolated"},
        ),
    )


def make_episode(
    reward: float = 5.0,
    steps: int = 3,
    terminated: bool = True,
    truncated: bool = False,
):
    return FakeEpisodeResult(
        total_reward=reward,
        step_count=steps,
        terminated=terminated,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# CONTRACT TESTS
# ---------------------------------------------------------------------------

def test_verifier_context_contract():
    task = make_task()
    scenario = {"scenario_id": "scenario-v4"}
    episode = make_episode()
    trajectory = ("step-0", "step-1", "step-2")

    context = VerifierContext(
        task_spec=task,
        scenario=scenario,
        episode_result=episode,
        trajectory=trajectory,
        metadata={"seed": 17},
    )

    assert context.task_spec is task
    assert context.scenario is scenario
    assert context.episode_result is episode
    assert context.trajectory == trajectory
    assert context.metadata["seed"] == 17


def test_verification_result_contract():
    result = VerificationResult(
        verified=True,
        verifier_id="contract-verifier",
        task_id="CONTRACT-TASK",
        episode=make_episode(),
        trajectory=("step",),
        reward={"valid": True},
        success={"success": True},
        diagnostics={"failed": False},
        metadata={"seed": 1},
    )

    assert result.verified is True
    assert result.passed is True
    assert result.verifier_id == "contract-verifier"
    assert result.task_id == "CONTRACT-TASK"
    assert result.episode is not None
    assert result.trajectory == ("step",)
    assert result.reward["valid"] is True
    assert result.success["success"] is True
    assert result.diagnostics["failed"] is False


def test_verification_result_is_immutable():
    result = VerificationResult(
        verified=True,
        verifier_id="immutable-verifier",
        task_id="IMMUTABLE-TASK",
        episode=make_episode(),
        trajectory=(),
        reward={},
        success={},
        diagnostics={},
    )

    with pytest.raises(Exception):
        result.verified = False


# ---------------------------------------------------------------------------
# EXECUTOR TESTS
# ---------------------------------------------------------------------------

def test_executor_resolves_and_executes_registered_verifier():
    verifier = RecordingVerifier()
    registry = VerifierRegistry()
    registry.register("v4-verifier", verifier)

    episode = make_episode()
    trajectory = ("step-0", "step-1", "step-2")

    result = VerifierExecutor(
        registry=registry,
    ).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "s1"},
        episode_result=episode,
        trajectory=trajectory,
        metadata={"seed": 99},
    )

    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert len(verifier.contexts) == 1


def test_executor_passes_exact_runtime_objects():
    verifier = RecordingVerifier()
    registry = VerifierRegistry()
    registry.register("v4-verifier", verifier)

    task = make_task()
    scenario = {"scenario_id": "exact"}
    episode = make_episode()
    trajectory = ("A", "B")

    VerifierExecutor(registry=registry).execute(
        task_spec=task,
        scenario=scenario,
        episode_result=episode,
        trajectory=trajectory,
    )

    context = verifier.contexts[0]

    assert context.task_spec is task
    assert context.scenario is scenario
    assert context.episode_result is episode
    assert context.trajectory is trajectory


def test_executor_rejects_missing_task_spec():
    registry = VerifierRegistry()
    registry.register("v4-verifier", RecordingVerifier())

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=None,
            scenario={},
            episode_result=make_episode(),
            trajectory=(),
        )


def test_executor_rejects_missing_scenario():
    registry = VerifierRegistry()
    registry.register("v4-verifier", RecordingVerifier())

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario=None,
            episode_result=make_episode(),
            trajectory=(),
        )


def test_executor_rejects_missing_episode_result():
    registry = VerifierRegistry()
    registry.register("v4-verifier", RecordingVerifier())

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario={},
            episode_result=None,
            trajectory=(),
        )


def test_executor_rejects_missing_trajectory():
    registry = VerifierRegistry()
    registry.register("v4-verifier", RecordingVerifier())

    with pytest.raises(ValueError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(),
            scenario={},
            episode_result=make_episode(),
            trajectory=None,
        )


def test_executor_rejects_non_structured_verifier_result():
    class BadVerifier:
        verifier_id = "bad-result"

        def verify(self, context):
            return True

    registry = VerifierRegistry()
    registry.register("bad-result", BadVerifier())

    with pytest.raises(TypeError):
        VerifierExecutor(registry=registry).execute(
            task_spec=make_task(verifier_id="bad-result"),
            scenario={},
            episode_result=make_episode(),
            trajectory=(),
        )


# ---------------------------------------------------------------------------
# TASKSPEC -> VERIFIERSPEC INTEGRATION
# ---------------------------------------------------------------------------

def test_task_spec_verifier_spec_selects_verifier():
    verifier_a = RecordingVerifier(verifier_id="verifier-a")
    verifier_b = RecordingVerifier(verifier_id="verifier-b")

    registry = VerifierRegistry()
    registry.register("verifier-a", verifier_a)
    registry.register("verifier-b", verifier_b)

    task = make_task(verifier_id="verifier-b")

    result = VerifierExecutor(registry=registry).execute(
        task_spec=task,
        scenario={"scenario_id": "selection"},
        episode_result=make_episode(),
        trajectory=("step",),
    )

    assert result.verifier_id == "verifier-b"
    assert len(verifier_a.contexts) == 0
    assert len(verifier_b.contexts) == 1


def test_task_spec_verifier_id_mismatch_is_rejected():
    verifier = RecordingVerifier(verifier_id="registered-id")
    registry = VerifierRegistry()
    registry.register("registered-id", verifier)

    task = make_task(verifier_id="different-id")

    with pytest.raises(KeyError):
        VerifierExecutor(registry=registry).execute(
            task_spec=task,
            scenario={},
            episode_result=make_episode(),
            trajectory=(),
        )


# ---------------------------------------------------------------------------
# EPISODERESULT -> TRAJECTORY INTEGRATION
# ---------------------------------------------------------------------------

def test_episode_and_trajectory_are_verified_independently():
    verifier = RecordingVerifier()
    registry = VerifierRegistry()
    registry.register("v4-verifier", verifier)

    episode = make_episode(steps=3)
    trajectory = ("step-0", "step-1", "step-2")

    result = VerifierExecutor(registry=registry).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "trajectory"},
        episode_result=episode,
        trajectory=trajectory,
    )

    assert result.metadata["episode_verification"].valid is True
    assert result.metadata["trajectory_verification"].valid is True
    assert result.metadata["reward_verification"].valid is True
    assert result.metadata["success_verification"].success is True


def test_episode_step_count_and_trajectory_length_remain_distinct():
    verifier = RecordingVerifier()
    registry = VerifierRegistry()
    registry.register("v4-verifier", verifier)

    episode = make_episode(steps=3)
    trajectory = ("step-0", "step-1")

    result = VerifierExecutor(registry=registry).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "different-length"},
        episode_result=episode,
        trajectory=trajectory,
    )

    assert result.metadata["episode_verification"].step_count == 3
    assert result.metadata["trajectory_verification"].length == 2


# ---------------------------------------------------------------------------
# COMPONENT VERIFIER CONTRACTS
# ---------------------------------------------------------------------------

def test_episode_verification_contract():
    result = verify_episode(
        step_count=4,
        terminated=True,
        truncated=False,
    )

    assert isinstance(result, EpisodeVerification)
    assert result.valid is True
    assert result.step_count == 4


def test_trajectory_verification_contract():
    result = verify_trajectory(
        ("step-0", "step-1", "step-2")
    )

    assert isinstance(result, TrajectoryVerification)
    assert result.valid is True
    assert result.length == 3


def test_reward_verification_contract():
    result = verify_reward(10.5)

    assert isinstance(result, RewardVerification)
    assert result.valid is True
    assert result.value == 10.5


def test_success_aggregation_contract():
    result = verify_success(
        episode_valid=True,
        reward_valid=True,
        trajectory_valid=True,
    )

    assert isinstance(result, type(verify_success(
        episode_valid=True,
        reward_valid=True,
        trajectory_valid=True,
    )))
    assert result.success is True


def test_failure_diagnostics_contract():
    result = FailureDiagnostics(
        failed=True,
        failures=("episode", "reward"),
    )

    assert result.failed is True
    assert result.failures == ("episode", "reward")


# ---------------------------------------------------------------------------
# VERIFIER ISOLATION
# ---------------------------------------------------------------------------

def test_verifier_receives_context_but_not_internal_registry():
    verifier = RecordingVerifier()
    registry = VerifierRegistry()
    registry.register("v4-verifier", verifier)

    VerifierExecutor(registry=registry).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "isolated"},
        episode_result=make_episode(),
        trajectory=("step",),
    )

    context = verifier.contexts[0]

    assert context.task_spec is not registry
    assert not hasattr(context, "registry")
    assert not hasattr(context, "_verifiers")


def test_verifier_context_does_not_expose_registry():
    verifier = RecordingVerifier()
    registry = VerifierRegistry()
    registry.register("v4-verifier", verifier)

    VerifierExecutor(registry=registry).execute(
        task_spec=make_task(),
        scenario={"scenario_id": "isolated"},
        episode_result=make_episode(),
        trajectory=("step",),
    )

    context = verifier.contexts[0]

    assert "registry" not in context.metadata
    assert "_verifiers" not in context.metadata


def test_verifier_only_gets_declared_context_objects():
    verifier = RecordingVerifier()
    registry = VerifierRegistry()
    registry.register("v4-verifier", verifier)

    task = make_task()
    scenario = {"scenario_id": "declared"}
    episode = make_episode()
    trajectory = ("step",)

    VerifierExecutor(registry=registry).execute(
        task_spec=task,
        scenario=scenario,
        episode_result=episode,
        trajectory=trajectory,
        metadata={"public": "value"},
    )

    context = verifier.contexts[0]

    assert set(vars(context).keys()) == {
        "task_spec",
        "scenario",
        "episode_result",
        "trajectory",
        "metadata",
    }

    assert context.metadata == {"public": "value"}
