from __future__ import annotations

from agentforge.schema import (
    SchemaValidationError,
    validate_named_schema,
)


def valid_reward():
    return {
        "components": [
            {
                "name": "progress",
                "value": 1.0,
                "weight": 1.0,
            }
        ],
        "total": 1.0,
    }


def valid_trajectory():
    return {
        "episode_id": "episode-1",
        "task_id": "TASK-001",
        "scenario_id": "scenario-1",
        "seed": 42,
        "steps": [
            {
                "episode_id": "episode-1",
                "task_id": "TASK-001",
                "scenario_id": "scenario-1",
                "seed": 42,
                "step_index": 0,
                "observation": {"state": "start"},
                "action": "continue",
                "reward_total": 1.0,
                "reward_components": valid_reward(),
                "terminated": False,
                "truncated": False,
                "info": {},
            }
        ],
    }


def test_runtime_reward_schema_validation_passes():
    validate_named_schema(valid_reward(), "reward.schema.json")


def test_runtime_trajectory_schema_validation_passes():
    validate_named_schema(valid_trajectory(), "trajectory.schema.json")


def test_runtime_validation_rejects_missing_required_field():
    invalid = valid_trajectory()
    del invalid["episode_id"]

    try:
        validate_named_schema(invalid, "trajectory.schema.json")
    except SchemaValidationError:
        pass
    else:
        raise AssertionError("Expected SchemaValidationError")


def test_runtime_validation_rejects_unknown_trajectory_field():
    invalid = valid_trajectory()
    invalid["unexpected_field"] = True

    try:
        validate_named_schema(invalid, "trajectory.schema.json")
    except SchemaValidationError:
        pass
    else:
        raise AssertionError("Expected SchemaValidationError")


def test_runtime_validation_resolves_cross_schema_reward_reference():
    trajectory = valid_trajectory()
    trajectory["steps"][0]["reward_components"]["unexpected"] = True

    try:
        validate_named_schema(trajectory, "trajectory.schema.json")
    except SchemaValidationError:
        pass
    else:
        raise AssertionError(
            "Expected nested reward.schema.json validation to reject the instance"
        )
