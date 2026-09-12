from __future__ import annotations

import ast
from pathlib import Path

from evaluator.harness import EvaluatorHarness
from evaluator.hidden_scenarios import load_hidden_scenarios
from evaluator.runner import run_hidden_evaluation


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PACKAGE = ROOT / "cuda_kernel_optimization_v1"


def test_hidden_scenarios_exist_only_in_evaluator() -> None:
    hidden_ids = {
        scenario.scenario_id
        for scenario in load_hidden_scenarios()
    }

    assert hidden_ids

    public_python = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PUBLIC_PACKAGE.glob("*.py")
    )

    for hidden_id in hidden_ids:
        assert hidden_id not in public_python


def test_public_taskset_does_not_import_evaluator() -> None:
    taskset_source = (
        PUBLIC_PACKAGE / "taskset.py"
    ).read_text(encoding="utf-8")

    assert "evaluator" not in taskset_source
    assert "hidden_scenarios" not in taskset_source


def test_agent_facing_package_has_no_evaluator_imports() -> None:
    forbidden_imports = {
        "evaluator",
        "hidden_scenarios",
    }

    for path in PUBLIC_PACKAGE.glob("*.py"):
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {
                    alias.name.split(".")[0]
                    for alias in node.names
                }
                assert not forbidden_imports.intersection(imported)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".")[0] not in forbidden_imports


def test_evaluator_harness_can_execute() -> None:
    harness = EvaluatorHarness()

    result = harness.evaluate(
        task_id="cuda_kernel_optimization_v1",
        scenario_id="evaluator-test",
        seed=1234,
        actions=[
            "inspect",
            "modify_tiling",
            "compile",
            "check_correctness",
            "benchmark",
            "analyze",
            "modify_tiling",
            "compile",
            "check_correctness",
            "benchmark",
            "analyze",
            "final_verify",
        ],
    )

    assert result.task_id == "cuda_kernel_optimization_v1"
    assert result.scenario_id == "evaluator-test"
    assert result.verifier_passed
    assert result.oracle_passed
    assert result.success
    assert result.score == 1.0


def test_hidden_runner_uses_hidden_scenarios() -> None:
    results = run_hidden_evaluation(
        actions=[
            "inspect",
            "modify_tiling",
            "compile",
            "check_correctness",
            "benchmark",
            "analyze",
            "modify_tiling",
            "compile",
            "check_correctness",
            "benchmark",
            "analyze",
            "final_verify",
        ]
    )

    hidden_ids = {
        scenario.scenario_id
        for scenario in load_hidden_scenarios()
    }

    assert results
    assert {result.scenario_id for result in results} == hidden_ids


def test_hidden_evaluation_does_not_modify_public_taskset() -> None:
    from cuda_kernel_optimization_v1.taskset import (
        CudaKernelOptimizationTaskset,
    )
    from verifiers.v1 import TasksetConfig

    taskset = CudaKernelOptimizationTaskset(
        TasksetConfig(id="cuda_kernel_optimization_v1")
    )

    public_ids_before = [
        task.data.scenario_id
        for task in taskset.load()
    ]

    run_hidden_evaluation(
        actions=[
            "inspect",
            "modify_tiling",
            "compile",
            "check_correctness",
            "benchmark",
            "analyze",
            "modify_tiling",
            "compile",
            "check_correctness",
            "benchmark",
            "analyze",
            "final_verify",
        ]
    )

    public_ids_after = [
        task.data.scenario_id
        for task in taskset.load()
    ]

    assert public_ids_before == public_ids_after
    assert all(
        "hidden" not in scenario_id
        for scenario_id in public_ids_after
    )
