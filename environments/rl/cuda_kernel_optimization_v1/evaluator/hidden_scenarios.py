from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HiddenKernelScenario:
    scenario_id: str
    seed: int
    matrix_size: int
    baseline_ms: float
    target_speedup: float


HIDDEN_SCENARIOS: tuple[HiddenKernelScenario, ...] = (
    HiddenKernelScenario(
        scenario_id="cuda-v1-hidden-001",
        seed=9001,
        matrix_size=768,
        baseline_ms=15.40,
        target_speedup=1.25,
    ),
)


def load_hidden_scenarios() -> tuple[HiddenKernelScenario, ...]:
    return HIDDEN_SCENARIOS
