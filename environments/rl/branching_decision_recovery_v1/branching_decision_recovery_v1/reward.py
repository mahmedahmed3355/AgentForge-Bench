from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardContract:
    useful_information: float = 0.0
    causal_discovery: float = 0.0
    valid_transition: float = 0.0
    recovery_progress: float = 0.0
    successful_replan: float = 0.0

    redundant_action: float = 0.0
    invalid_action: float = -0.5
    wrong_decision: float = -1.0
    bad_recovery: float = -2.0

    final_success: float = 10.0
    terminal_failure: float = -5.0
