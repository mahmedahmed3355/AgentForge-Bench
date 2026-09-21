
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from agentforge.runtime.evaluation_runner import EvaluationRunner
from agentforge.verifier import (
    VerificationResult,
    VerifierExecutor,
    VerifierRegistry,
)


@dataclass
class FakeEnvironment:
    def __init__(self):
        self.closed = False
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return "initial"

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= 2
        return (
            f"observation-{self.current_step}",
            1.0,
            terminated,
            False,
            {},
        )

    def close(self):
        self.closed = True


class FakeEnvironmentFactory:
    def __init__(self):
        self.environments = []

    def create(self, task_id):
        environment = FakeEnvironment()
        self.environments.append(environment)
        return environment


class FakeVerifier:
    verifier_id = "evaluation-verifier"

    def verify(self, context):
        return VerificationResult(
            verified=True,
            verifier_id=self.verifier_id,
            task_id=context.task_spec.identity.task_id,
            episode=context.episode_result,
            trajectory=context.trajectory,
            reward={"total": context.episode_result.total_reward},
            success={"success": True},
            diagnostics={"failed": False},
            metadata={"source": "evaluation"},
        )


def make_task(task_id="EVAL-TASK"):
    return SimpleNamespace(
        identity=SimpleNamespace(task_id=task_id),
        verifier=SimpleNamespace(
            verifier_id="evaluation-verifier",
            implementation=None,
            configuration={},
        ),
    )


def test_evaluation_remains_backward_compatible_without_verifier():
    factory = FakeEnvironmentFactory()

    runner = EvaluationRunner(factory)

    result = runner.run(
        "EVAL-TASK",
        [
            ("A",),
            ("A", "B"),
        ],
    )

    assert result.episode_count == 2
    assert len(result.episodes) == 2
    assert result.total_reward == 3.0
    assert result.verification_results == ()


def test_evaluation_runs_verifier_for_each_episode(monkeypatch):
    factory = FakeEnvironmentFactory()

    registry = VerifierRegistry()
    registry.register(
        "evaluation-verifier",
        FakeVerifier(),
    )

    executor = VerifierExecutor(
        registry=registry,
    )

    class FakeEpisodeRunner:
        def __init__(self, environment):
            self.environment = environment

        def run(self, actions):
            from agentforge.runtime.episode_runner import EpisodeResult

            return EpisodeResult(
                observation={"done": True},
                total_reward=float(len(actions)),
                step_count=len(actions),
                terminated=True,
                truncated=False,
                trajectory=tuple(actions),
            )

    monkeypatch.setattr(
        "agentforge.runtime.evaluation_runner.EpisodeRunner",
        FakeEpisodeRunner,
    )

    runner = EvaluationRunner(
        factory,
        verifier_executor=executor,
        task_spec_resolver=lambda task_id: make_task(task_id),
    )

    result = runner.run(
        "EVAL-TASK",
        [
            ("A",),
            ("A", "B"),
            ("A", "B", "C"),
        ],
    )

    assert result.episode_count == 3
    assert len(result.verification_results) == 3

    for verification in result.verification_results:
        assert isinstance(verification, VerificationResult)
        assert verification.verified is True
        assert verification.task_id == "EVAL-TASK"


def test_episode_result_and_trajectory_reach_verifier(monkeypatch):
    factory = FakeEnvironmentFactory()

    captured = []

    class CapturingVerifier:
        verifier_id = "evaluation-verifier"

        def verify(self, context):
            captured.append(context)

            return VerificationResult(
                verified=True,
                verifier_id=self.verifier_id,
                task_id=context.task_spec.identity.task_id,
                episode=context.episode_result,
                trajectory=context.trajectory,
                reward={},
                success={},
                diagnostics={},
            )

    registry = VerifierRegistry()
    registry.register(
        "evaluation-verifier",
        CapturingVerifier(),
    )

    executor = VerifierExecutor(
        registry=registry,
    )

    class FakeEpisodeRunner:
        def __init__(self, environment):
            self.environment = environment

        def run(self, actions):
            from agentforge.runtime.episode_runner import EpisodeResult

            return EpisodeResult(
                observation={"done": True},
                total_reward=float(len(actions)),
                step_count=len(actions),
                terminated=True,
                truncated=False,
                trajectory=tuple(actions),
            )

    monkeypatch.setattr(
        "agentforge.runtime.evaluation_runner.EpisodeRunner",
        FakeEpisodeRunner,
    )

    runner = EvaluationRunner(
        factory,
        verifier_executor=executor,
        task_spec_resolver=lambda task_id: make_task(task_id),
    )

    result = runner.run(
        "CAPTURE-TASK",
        [
            ("A", "B"),
        ],
    )

    assert len(captured) == 1
    assert captured[0].task_spec.identity.task_id == "CAPTURE-TASK"
    assert captured[0].episode_result is result.episodes[0]
    assert captured[0].trajectory == result.episodes[0].trajectory


def test_verification_results_preserve_episode_order(monkeypatch):
    factory = FakeEnvironmentFactory()

    registry = VerifierRegistry()
    registry.register(
        "evaluation-verifier",
        FakeVerifier(),
    )

    executor = VerifierExecutor(
        registry=registry,
    )

    class FakeEpisodeRunner:
        def __init__(self, environment):
            self.environment = environment

        def run(self, actions):
            from agentforge.runtime.episode_runner import EpisodeResult

            return EpisodeResult(
                observation={"done": True},
                total_reward=float(len(actions)),
                step_count=len(actions),
                terminated=True,
                truncated=False,
                trajectory=tuple(actions),
            )

    monkeypatch.setattr(
        "agentforge.runtime.evaluation_runner.EpisodeRunner",
        FakeEpisodeRunner,
    )

    runner = EvaluationRunner(
        factory,
        verifier_executor=executor,
        task_spec_resolver=lambda task_id: make_task(task_id),
    )

    result = runner.run(
        "ORDER-TASK",
        [
            ("A",),
            ("A", "B"),
            ("A", "B", "C"),
        ],
    )

    assert len(result.episodes) == 3
    assert len(result.verification_results) == 3

    for episode, verification in zip(
        result.episodes,
        result.verification_results,
    ):
        assert verification.episode is episode
        assert verification.trajectory == episode.trajectory


def test_scenario_resolver_receives_episode_result(monkeypatch):
    factory = FakeEnvironmentFactory()

    registry = VerifierRegistry()
    registry.register(
        "evaluation-verifier",
        FakeVerifier(),
    )

    executor = VerifierExecutor(
        registry=registry,
    )

    captured_scenarios = []

    def scenario_resolver(task_id, episode_index, episode_result):
        scenario = {
            "task_id": task_id,
            "episode_index": episode_index,
            "step_count": episode_result.step_count,
        }
        captured_scenarios.append(scenario)
        return scenario

    class FakeEpisodeRunner:
        def __init__(self, environment):
            self.environment = environment

        def run(self, actions):
            from agentforge.runtime.episode_runner import EpisodeResult

            return EpisodeResult(
                observation={"done": True},
                total_reward=float(len(actions)),
                step_count=len(actions),
                terminated=True,
                truncated=False,
                trajectory=tuple(actions),
            )

    monkeypatch.setattr(
        "agentforge.runtime.evaluation_runner.EpisodeRunner",
        FakeEpisodeRunner,
    )

    runner = EvaluationRunner(
        factory,
        verifier_executor=executor,
        task_spec_resolver=lambda task_id: make_task(task_id),
        scenario_resolver=scenario_resolver,
    )

    runner.run(
        "SCENARIO-TASK",
        [
            ("A",),
            ("A", "B"),
        ],
    )

    assert captured_scenarios == [
        {
            "task_id": "SCENARIO-TASK",
            "episode_index": 0,
            "step_count": 1,
        },
        {
            "task_id": "SCENARIO-TASK",
            "episode_index": 1,
            "step_count": 2,
        },
    ]
