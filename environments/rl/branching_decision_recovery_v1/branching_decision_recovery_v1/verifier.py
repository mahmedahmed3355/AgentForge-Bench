from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision_graph import DecisionBranch, DecisionStrategy


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    terminal: bool
    success: bool
    reason: str


class BranchingDecisionVerifier:
    """Independent terminal-state verifier.

    This verifier does not call the oracle and does not trust reward.
    """

    def verify(self, state: Any) -> VerificationResult:
        if not state.terminal:
            return VerificationResult(
                verified=False,
                terminal=False,
                success=False,
                reason="episode is not terminal",
            )

        if not state.success:
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="terminal episode reports failure",
            )

        if state.failure_detected:
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="active failure remains",
            )

        if state.recovery_required:
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="recovery remains required",
            )

        if state.replanning_required:
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="replanning remains required",
            )

        if state.selected_branch not in {
            item.value for item in DecisionBranch
        }:
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="invalid or missing branch",
            )

        if state.selected_strategy not in {
            item.value for item in DecisionStrategy
        }:
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="invalid or missing strategy",
            )

        if state.validation_status != "globally_validated":
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="global validation not completed",
            )

        if state.logical_stage < 10:
            return VerificationResult(
                verified=False,
                terminal=True,
                success=False,
                reason="insufficient logical progress",
            )

        return VerificationResult(
            verified=True,
            terminal=True,
            success=True,
            reason="terminal state satisfies independent verifier contract",
        )
