from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf

from cuda_kernel_optimization_v1.scenarios import (
    load_public_scenarios,
)
from cuda_kernel_optimization_v1.taskset import (
    CudaKernelOptimizationTaskset,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "cuda_kernel_optimization_v1"
EVALUATOR_ROOT = ROOT / "evaluator"


def _load_hidden_scenarios():
    from evaluator.hidden_scenarios import load_hidden_scenarios

    return load_hidden_scenarios()


def test_public_scenarios_do_not_include_hidden_scenarios():
    public_ids = {
        scenario.scenario_id
        for scenario in load_public_scenarios()
    }

    hidden_ids = {
        scenario.scenario_id
        for scenario in _load_hidden_scenarios()
    }

    assert public_ids
    assert hidden_ids
    assert public_ids.isdisjoint(hidden_ids)


def test_public_taskset_contains_no_hidden_scenario():
    taskset = CudaKernelOptimizationTaskset(
        config=vf.TasksetConfig(
            id="cuda_kernel_optimization_v1"
        )
    )

    tasks = taskset.load()

    task_ids = {
        task.data.scenario_id
        for task in tasks
    }

    hidden_ids = {
        scenario.scenario_id
        for scenario in _load_hidden_scenarios()
    }

    assert task_ids
    assert task_ids.isdisjoint(hidden_ids)


def test_public_scenario_module_contains_no_hidden_data():
    source = (
        PACKAGE_ROOT / "scenarios.py"
    ).read_text()

    forbidden = [
        "cuda-v1-hidden-001",
        "9001",
        "768",
        "15.40",
        "1.25",
        "HIDDEN_SCENARIOS",
        "load_hidden_scenarios",
    ]

    found = [
        marker
        for marker in forbidden
        if marker in source
    ]

    assert found == []


def test_taskset_source_contains_no_hidden_data():
    source = (
        PACKAGE_ROOT / "taskset.py"
    ).read_text()

    forbidden = [
        "cuda-v1-hidden-001",
        "9001",
        "768",
        "15.40",
        "1.25",
        "HIDDEN_SCENARIOS",
        "load_hidden_scenarios",
    ]

    found = [
        marker
        for marker in forbidden
        if marker in source
    ]

    assert found == []


def test_agent_package_does_not_import_evaluator():
    for path in PACKAGE_ROOT.glob("*.py"):
        source = path.read_text()

        assert "from evaluator" not in source
        assert "import evaluator" not in source


def test_hidden_data_exists_only_in_evaluator_area():
    hidden_source = (
        EVALUATOR_ROOT / "hidden_scenarios.py"
    ).read_text()

    assert "cuda-v1-hidden-001" in hidden_source
    assert "9001" in hidden_source

    for path in PACKAGE_ROOT.glob("*.py"):
        source = path.read_text()

        assert "cuda-v1-hidden-001" not in source
        assert "9001" not in source


def test_prime_v1_public_taskset_stays_public_only():
    taskset = CudaKernelOptimizationTaskset(
        config=vf.TasksetConfig(
            id="cuda_kernel_optimization_v1"
        )
    )

    tasks = taskset.load()

    assert len(tasks) == 3

    assert [
        task.data.scenario_id
        for task in tasks
    ] == [
        "cuda-v1-train-001",
        "cuda-v1-train-002",
        "cuda-v1-eval-001",
    ]
