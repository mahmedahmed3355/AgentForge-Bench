from __future__ import annotations

from pydantic import Field

from verifiers.v1.state import State


class CudaPrimeState(State):
    """Prime V1 rollout state for the CUDA optimization toolset.

    This is intentionally a transport state, not a replacement for the
    native CUDA environment state.
    """

    seed: int | None = None
    logical_stage: int = 0
    action_count: int = 0
    inspected: bool = False
    candidate_modified: bool = False
    compilation_ok: bool = False
    correctness_ok: bool = False
    benchmark_completed: bool = False
    analysis_completed: bool = False
    performance_improved: bool = False
    regression_detected: bool = False
    recovery_count: int = 0
    terminal: bool = False
    success: bool = False
    finalized: bool = False
    last_failure: str | None = None
    last_observation: str = ""
    history: list[str] = Field(default_factory=list)
