from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class VerifierContext:
    task_spec: Any
    scenario: Any
    episode_result: Any
    trajectory: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    verifier_id: str
    task_id: str
    episode: Any
    trajectory: Any
    reward: Any
    success: Any
    diagnostics: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verified


class BaseVerifier(Protocol):
    verifier_id: str

    def verify(self, context: VerifierContext) -> VerificationResult:
        ...
