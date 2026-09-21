from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas"


class SchemaValidationError(ValueError):
    """Raised when an instance does not satisfy a canonical schema."""


def load_schema(name: str) -> dict[str, Any]:
    """Load one canonical JSON schema by filename."""
    path = _SCHEMA_DIR / name

    if not path.is_file():
        raise FileNotFoundError(f"Schema not found: {name}")

    return json.loads(path.read_text())


def validate_instance(
    instance: Any,
    schema: dict[str, Any],
) -> None:
    """Validate an instance against a Draft 2020-12 schema."""
    resources = {}

    for schema_path in _SCHEMA_DIR.glob("*.schema.json"):
        document = json.loads(schema_path.read_text())
        schema_id = document.get("$id")

        if schema_id:
            resources[schema_id] = Resource.from_contents(document)

    registry = Registry().with_resources(resources.items())

    validator = Draft202012Validator(
        schema,
        registry=registry,
    )

    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: list(error.path),
    )

    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path)
        if not location:
            location = "<root>"

        raise SchemaValidationError(
            f"Schema validation failed at {location}: {first.message}"
        )


def validate_named_schema(
    instance: Any,
    schema_name: str,
) -> None:
    """Validate an instance against a canonical repository schema."""
    validate_instance(instance, load_schema(schema_name))
