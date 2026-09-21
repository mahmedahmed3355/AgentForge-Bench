from __future__ import annotations

from typing import Any, Callable, Mapping

from .contracts import BaseVerifier, VerificationResult, VerifierContext


class VerifierRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, BaseVerifier | Callable[[], BaseVerifier]] = {}

    def register(
        self,
        verifier_id: str,
        verifier: BaseVerifier | Callable[[], BaseVerifier],
    ) -> None:
        if not verifier_id or not verifier_id.strip():
            raise ValueError("verifier_id must not be empty")

        if verifier_id in self._verifiers:
            raise ValueError(f"Verifier already registered: {verifier_id}")

        self._verifiers[verifier_id] = verifier

    def resolve(self, verifier_id: str) -> BaseVerifier:
        if verifier_id not in self._verifiers:
            raise KeyError(f"Unknown verifier: {verifier_id}")

        registered = self._verifiers[verifier_id]

        verifier = registered() if callable(registered) and not hasattr(
            registered, "verify"
        ) else registered

        if not hasattr(verifier, "verify"):
            raise TypeError(
                f"Registered verifier '{verifier_id}' does not implement verify()"
            )

        return verifier


class VerifierExecutor:
    def __init__(
        self,
        verifier: BaseVerifier | None = None,
        *,
        registry: VerifierRegistry | None = None,
        verifier_id: str | None = None,
    ) -> None:
        self.registry = registry
        self.verifier = verifier
        self.verifier_id = verifier_id

        if self.verifier is not None and self.verifier_id is None:
            self.verifier_id = getattr(
                self.verifier,
                "verifier_id",
                type(self.verifier).__name__,
            )

    def resolve(self, verifier_spec: Any) -> BaseVerifier:
        if verifier_spec is None:
            raise ValueError("verifier_spec is required")

        spec_id = getattr(verifier_spec, "verifier_id", None)

        if not spec_id:
            raise ValueError("VerifierSpec.verifier_id is required")

        if self.registry is None:
            if self.verifier is not None:
                if self.verifier_id != spec_id:
                    raise ValueError(
                        "Configured verifier does not match VerifierSpec.verifier_id"
                    )
                return self.verifier

            raise RuntimeError(
                "No VerifierRegistry or configured verifier is available"
            )

        return self.registry.resolve(spec_id)

    def execute(
        self,
        *,
        task_spec: Any,
        scenario: Any,
        episode_result: Any,
        trajectory: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        if task_spec is None:
            raise ValueError("task_spec is required")

        if scenario is None:
            raise ValueError("scenario is required")

        if episode_result is None:
            raise ValueError("episode_result is required")

        if trajectory is None:
            raise ValueError("trajectory is required")

        verifier_spec = getattr(task_spec, "verifier", None)

        if verifier_spec is None:
            raise ValueError("TaskSpec.verifier is required")

        verifier = self.resolve(verifier_spec)

        context = VerifierContext(
            task_spec=task_spec,
            scenario=scenario,
            episode_result=episode_result,
            trajectory=trajectory,
            metadata=dict(metadata or {}),
        )

        from .diagnostics import diagnose_failures
        from .episode import verify_episode
        from .reward import verify_reward
        from .success import verify_success
        from .trajectory import verify_trajectory

        episode_verification = verify_episode(
            step_count=episode_result.step_count,
            terminated=episode_result.terminated,
            truncated=episode_result.truncated,
        )

        trajectory_verification = verify_trajectory(trajectory)

        reward_verification = verify_reward(episode_result.total_reward)

        success_verification = verify_success(
            episode_valid=episode_verification.valid,
            reward_valid=reward_verification.valid,
            trajectory_valid=trajectory_verification.valid,
        )

        diagnostics = diagnose_failures(
            (
                ("episode", episode_verification.valid),
                ("trajectory", trajectory_verification.valid),
                ("reward", reward_verification.valid),
                ("success", success_verification.success),
            )
        )

        result = verifier.verify(context)

        if not isinstance(result, VerificationResult):
            raise TypeError(
                "Verifier must return agentforge.verifier.contracts.VerificationResult"
            )

        expected_task_id = getattr(
            getattr(task_spec, "identity", None),
            "task_id",
            getattr(task_spec, "task_id", None),
        )

        if expected_task_id is not None and result.task_id != expected_task_id:
            raise ValueError(
                "VerificationResult.task_id does not match TaskSpec task_id"
            )

        if result.verifier_id != verifier_spec.verifier_id:
            raise ValueError(
                "VerificationResult.verifier_id does not match VerifierSpec.verifier_id"
            )

        integrated_metadata = dict(result.metadata)
        integrated_metadata.update(
            {
                "episode_verification": episode_verification,
                "trajectory_verification": trajectory_verification,
                "reward_verification": reward_verification,
                "success_verification": success_verification,
                "diagnostics": diagnostics,
            }
        )

        return VerificationResult(
            verified=(
                result.verified
                and episode_verification.valid
                and trajectory_verification.valid
                and reward_verification.valid
                and success_verification.success
                and not diagnostics.failed
            ),
            verifier_id=result.verifier_id,
            task_id=result.task_id,
            episode=result.episode,
            trajectory=result.trajectory,
            reward=reward_verification,
            success=success_verification,
            diagnostics=diagnostics,
            metadata=integrated_metadata,
        )
