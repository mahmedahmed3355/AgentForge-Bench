"""Structured framework errors."""


class FrameworkError(Exception):
    """Base class for framework errors."""


class InvalidActionError(FrameworkError):
    """Raised when an agent action violates the action contract."""


class EnvironmentStateError(FrameworkError):
    """Raised when an environment lifecycle rule is violated."""


class TaskConfigurationError(FrameworkError):
    """Raised when task configuration is invalid."""


class ScenarioGenerationError(FrameworkError):
    """Raised when scenario generation fails."""


class EvaluationError(FrameworkError):
    """Raised when evaluator-side processing fails."""
