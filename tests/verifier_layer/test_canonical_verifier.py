from __future__ import annotations

from dataclasses import dataclass

from agentforge.verifier import (
    BaseVerifier,
    VerificationResult,
    VerifierContext,
    VerifierExecutor,
)


@dataclass(frozen=True)
class FakeTask:
    task_id: str = "TEST-TASK"


@dataclass(frozen=True)
class FakeEpisode:
    total_reward: float = 1.0
    step_count: int = 1
    terminated: bool = True
    truncated: bool = False


class FakeVerifier:
    verifier_id = "fake-verifier"

    def verify(self, context: VerifierContext) -> VerificationResult:
        assert context.task_spec is not None
        assert context.scenario is not None
        assert context.episode_result is not None
        assert context.trajectory is not None

        return VerificationResult(
            verified=True,
            verifier_id=self.verifier_id,
            task_id=context.task_spec.task_id,
            episode=context.episode_result,
            trajectory=context.trajectory,
            reward={"total": context.episode_result.total_reward},
            success={"success": True},
            diagnostics={"failures": ()},
            metadata=context.metadata,
        )


def test_canonical_verifier_context_and_result():
    task = FakeTask()
    episode = FakeEpisode()
    trajectory = ("step-0",)

    context = VerifierContext(
        task_spec=task,
        scenario={"scenario_id": "scenario-1"},
        episode_result=episode,
        trajectory=trajectory,
    )

    assert context.task_spec is task
    assert context.episode_result is episode
    assert context.trajectory == trajectory

    result = FakeVerifier().verify(context)

    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert result.passed is True
    assert result.verifier_id == "fake-verifier"
    assert result.task_id == "TEST-TASK"


def test_verifier_executor_returns_structured_result():
    class Task:
        task_id = "EXEC-TASK"
        verifier = type(
            "VerifierSpec",
            (),
            {
                "verifier_id": "fake-verifier",
                "implementation": None,
                "configuration": {},
            },
        )()

    episode = FakeEpisode()

    result = VerifierExecutor(
        verifier=FakeVerifier(),
        verifier_id="fake-verifier",
    ).execute(
        task_spec=Task(),
        scenario={"scenario_id": "s1"},
        episode_result=episode,
        trajectory=("step-0",),
        metadata={"seed": 7},
    )

    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert result.task_id == "EXEC-TASK"
    assert result.metadata["seed"] == 7


def test_verifier_executor_rejects_boolean_result():
    class BadVerifier:
        verifier_id = "bad"

        def verify(self, context):
            return True

    class Task:
        task_id = "BAD-TASK"
        verifier = type(
            "VerifierSpec",
            (),
            {
                "verifier_id": "bad",
                "implementation": None,
                "configuration": {},
            },
        )()

    try:
        VerifierExecutor(
            verifier=BadVerifier(),
            verifier_id="bad",
        ).execute(
            task_spec=Task(),
            scenario={},
            episode_result=FakeEpisode(),
            trajectory=(),
        )
        raised = False
    except TypeError:
        raised = True

    assert raised is True


def test_verifier_protocol_shape_is_available():
    assert BaseVerifier is not None
