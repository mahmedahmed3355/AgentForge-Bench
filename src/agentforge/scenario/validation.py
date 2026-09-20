from __future__ import annotations

from .base import Scenario


class ScenarioValidationError(ValueError):
    """Raised when a scenario violates the framework contract."""


def validate_scenario(scenario: Scenario) -> None:
    """Validate the structural scenario contract."""

    if not isinstance(scenario, Scenario):
        raise ScenarioValidationError("scenario must be a Scenario")

    if not scenario.task_id:
        raise ScenarioValidationError("scenario task_id must be non-empty")

    if not scenario.scenario_id:
        raise ScenarioValidationError("scenario scenario_id must be non-empty")

    if not isinstance(scenario.seed, int):
        raise ScenarioValidationError("scenario seed must be an integer")

    if not isinstance(scenario.public, dict):
        raise ScenarioValidationError("scenario public data must be a dictionary")

    if not isinstance(scenario.hidden, dict):
        raise ScenarioValidationError("scenario hidden data must be a dictionary")

    if not isinstance(scenario.metadata, dict):
        raise ScenarioValidationError("scenario metadata must be a dictionary")
