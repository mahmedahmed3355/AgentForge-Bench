from __future__ import annotations

from verifiers.v1.state import State


class BranchingDecisionPrimeState(State):
    """Prime-facing state synchronized from the native environment."""

    scenario_id: str = ""
    seed: int | None = None
    logical_stage: int = 0
    step: int = 0
    terminal: bool = False
    success: bool = False
    last_observation: str = ""
