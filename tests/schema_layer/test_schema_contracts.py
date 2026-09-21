
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schemas"


EXPECTED = {
    "common.schema.json",
    "reward.schema.json",
    "trajectory.schema.json",
    "episode.schema.json",
    "evaluation.schema.json",
    "task.schema.json",
}


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def test_all_canonical_schemas_exist() -> None:
    actual = {p.name for p in SCHEMA_DIR.glob("*.schema.json")}
    assert EXPECTED.issubset(actual)


def test_all_schemas_are_draft_2020_12() -> None:
    for name in EXPECTED:
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "$id" in schema
        assert "title" in schema


def test_trajectory_schema_matches_canonical_contract() -> None:
    schema = load_schema("trajectory.schema.json")
    assert schema["required"] == [
        "episode_id",
        "task_id",
        "scenario_id",
        "seed",
        "steps",
    ]

    step = schema["$defs"]["trajectory_step"]
    assert step["required"] == [
        "episode_id",
        "task_id",
        "scenario_id",
        "seed",
        "step_index",
        "observation",
        "action",
        "reward_total",
        "reward_components",
        "terminated",
        "truncated",
        "info",
    ]


def test_episode_schema_matches_episode_result_contract() -> None:
    schema = load_schema("episode.schema.json")
    assert schema["required"] == [
        "observation",
        "total_reward",
        "step_count",
        "terminated",
        "truncated",
        "trajectory",
    ]


def test_evaluation_schema_matches_evaluation_result_contract() -> None:
    schema = load_schema("evaluation.schema.json")
    assert schema["required"] == [
        "episodes",
        "episode_count",
        "total_reward",
        "verification_results",
    ]


def test_task_schema_contains_full_task_contract() -> None:
    schema = load_schema("task.schema.json")

    assert schema["required"] == [
        "identity",
        "environment",
        "scenario",
        "actions",
        "observations",
        "reward",
        "termination",
        "success",
        "oracle",
        "verifier",
        "compatibility",
    ]

    identity = schema["$defs"]["task_identity"]
    assert {"task_id", "version", "domain"} <= set(identity["required"])

    verifier = schema["$defs"]["verifier_spec"]
    assert verifier["required"] == ["verifier_id"]

    compatibility = schema["$defs"]["compatibility_spec"]
    assert compatibility["required"] == ["contract_version"]


def test_reward_schema_matches_reward_breakdown_contract() -> None:
    schema = load_schema("reward.schema.json")

    assert schema["required"] == ["components", "total"]

    component = schema["$defs"]["reward_component"]
    assert component["required"] == ["name", "value"]


def test_strict_structural_objects_reject_unknown_fields() -> None:
    for name in [
        "reward.schema.json",
        "trajectory.schema.json",
        "episode.schema.json",
        "evaluation.schema.json",
        "task.schema.json",
    ]:
        schema = load_schema(name)
        assert schema.get("additionalProperties") is False


def test_jsonschema_library_validates_canonical_schemas() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    for name in EXPECTED:
        schema = load_schema(name)
        jsonschema.Draft202012Validator.check_schema(schema)

def test_task_objective_matches_python_contract() -> None:
    schema = load_schema("task.schema.json")
    objective = schema["properties"]["objective"]

    assert objective == {"type": "string"}
