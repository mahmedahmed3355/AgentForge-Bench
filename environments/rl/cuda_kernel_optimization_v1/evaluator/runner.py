from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .harness import EvaluatorHarness, EvaluationResult
from .hidden_scenarios import load_hidden_scenarios


def run_hidden_evaluation(
    *,
    actions: list[str],
) -> list[EvaluationResult]:
    """
    Run agent actions against evaluator-owned hidden scenarios.

    Hidden scenario definitions are loaded only inside evaluator code.
    """
    harness = EvaluatorHarness()
    results: list[EvaluationResult] = []

    for scenario in load_hidden_scenarios():
        result = harness.evaluate(
            task_id="cuda_kernel_optimization_v1",
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            actions=actions,
        )
        results.append(result)

    return results


def save_results(
    results: list[EvaluationResult],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "task_id": "cuda_kernel_optimization_v1",
        "num_scenarios": len(results),
        "results": [
            EvaluatorHarness.result_to_dict(result)
            for result in results
        ],
    }

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
