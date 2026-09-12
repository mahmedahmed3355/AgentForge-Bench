from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    seed: int
    matrix_size: int
    baseline_ms: float
    target_speedup: float


SCENARIOS = (
    Scenario(
        scenario_id="cuda-v1-train-001",
        split="train",
        seed=1001,
        matrix_size=256,
        baseline_ms=2.50,
        target_speedup=1.10,
    ),
    Scenario(
        scenario_id="cuda-v1-train-002",
        split="train",
        seed=1002,
        matrix_size=384,
        baseline_ms=4.20,
        target_speedup=1.15,
    ),
    Scenario(
        scenario_id="cuda-v1-eval-001",
        split="eval",
        seed=2001,
        matrix_size=512,
        baseline_ms=7.80,
        target_speedup=1.20,
    ),
    Scenario(
        scenario_id="cuda-v1-hidden-001",
        split="hidden",
        seed=9001,
        matrix_size=768,
        baseline_ms=15.40,
        target_speedup=1.25,
    ),
)
