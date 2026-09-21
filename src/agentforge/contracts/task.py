"""Canonical AgentForge task contracts.

The task contract describes benchmark-task identity and lifecycle
compatibility, while TaskSpec describes the complete declarative task
architecture.

Real task implementations are intentionally outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


class TaskEnvironment(Protocol):
    """Compatibility protocol for environments consumed by tasks.

    The canonical concrete environment implementation is AgentForgeEnv.
    This protocol exists only as a structural compatibility boundary.
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class TaskContract:
    """Stable identity and episode-boundary contract."""

    task_id: str
    version: str
    max_episode_steps: int

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty.")

        if not self.version:
            raise ValueError("version must not be empty.")

        if self.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive."
            )


@dataclass(frozen=True)
class TaskIdentity:
    """Canonical task identity."""

    task_id: str
    version: str
    domain: str
    split: str = "train"

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty.")

        if not self.version:
            raise ValueError("version must not be empty.")

        if not self.domain:
            raise ValueError("domain must not be empty.")

        if not self.split:
            raise ValueError("split must not be empty.")


@dataclass(frozen=True)
class EnvironmentBinding:
    """Binds a task specification to its canonical environment."""

    environment_id: str
    environment_version: str = "1"
    version: str = "1.0"
    max_episode_steps: int = 1

    def __post_init__(self) -> None:
        if not self.environment_id:
            raise ValueError(
                "environment_id must not be empty."
            )

        if not self.environment_version:
            raise ValueError(
                "environment_version must not be empty."
            )

        if self.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive."
            )


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative scenario definition."""

    scenario_id: str
    generator: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError(
                "scenario_id must not be empty."
            )


@dataclass(frozen=True)
class ActionSpec:
    """Action vocabulary exposed to an agent."""

    vocabulary: tuple[str, ...]
    schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocabulary:
            raise ValueError(
                "action vocabulary must not be empty."
            )

        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError(
                "action vocabulary must not contain duplicates."
            )


@dataclass(frozen=True)
class ObservationSpec:
    """Observation schema exposed to an agent."""

    schema: Mapping[str, Any]
    description: str = ""


@dataclass(frozen=True)
class RewardSpec:
    """Declarative reward specification."""

    kind: str = "scalar"
    specification: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError(
                "reward kind must not be empty."
            )


@dataclass(frozen=True)
class TerminationSpec:
    """Episode termination and truncation configuration."""

    termination_conditions: tuple[str, ...] = ()
    truncation_conditions: tuple[str, ...] = ()
    max_episode_steps: int | None = None

    def __post_init__(self) -> None:
        if (
            self.max_episode_steps is not None
            and self.max_episode_steps <= 0
        ):
            raise ValueError(
                "max_episode_steps must be positive when provided."
            )


@dataclass(frozen=True)
class SuccessSpec:
    """Task success criteria."""

    criteria: tuple[str, ...]
    success_reward: float | None = None

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError(
                "success criteria must not be empty."
            )


@dataclass(frozen=True)
class OracleSpec:
    """Reference/oracle configuration."""

    oracle_id: str
    implementation: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.oracle_id:
            raise ValueError(
                "oracle_id must not be empty."
            )


@dataclass(frozen=True)
class VerifierSpec:
    """Verifier configuration."""

    verifier_id: str
    implementation: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError(
                "verifier_id must not be empty."
            )


@dataclass(frozen=True)
class CompatibilitySpec:
    """Framework compatibility requirements."""

    contract_version: str
    gymnasium_version: str | None = None
    framework_version: str | None = None

    def __post_init__(self) -> None:
        if not self.contract_version:
            raise ValueError(
                "contract_version must not be empty."
            )


@dataclass(frozen=True)
class TaskSpec:
    """Complete canonical declarative benchmark-task specification.

    Conceptual architecture:

        TaskSpec
        ├── identity
        ├── environment
        ├── scenario
        ├── actions
        ├── observations
        ├── reward
        ├── termination
        ├── success
        ├── oracle
        ├── verifier
        └── compatibility

    This object defines task configuration only. It does not execute
    environments, generate scenarios, run oracles, or perform verification.
    """

    identity: TaskIdentity
    environment: EnvironmentBinding
    scenario: ScenarioSpec
    actions: ActionSpec
    observations: ObservationSpec
    reward: RewardSpec
    termination: TerminationSpec
    success: SuccessSpec
    oracle: OracleSpec
    verifier: VerifierSpec
    compatibility: CompatibilitySpec

    # Existing architecture fields retained as first-class task metadata.
    objective: str = ""
    difficulty: float = 0.0
    horizon: int | None = None
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.difficulty < 0:
            raise ValueError(
                "difficulty must be non-negative."
            )

        if self.horizon is not None and self.horizon <= 0:
            raise ValueError(
                "horizon must be positive when provided."
            )

        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError(
                "capabilities must not contain duplicates."
            )

    @property
    def task_id(self) -> str:
        """Compatibility accessor for the canonical task identity."""
        return self.identity.task_id

    @property
    def version(self) -> str:
        """Compatibility accessor for the canonical task identity."""
        return self.identity.version

    @property
    def domain(self) -> str:
        """Compatibility accessor for the canonical task identity."""
        return self.identity.domain

    @property
    def split(self) -> str:
        """Compatibility accessor for the canonical task identity."""
        return self.identity.split

    @property
    def max_episode_steps(self) -> int:
        """Canonical episode limit from environment/termination config."""
        if self.termination.max_episode_steps is not None:
            return self.termination.max_episode_steps
        return self.environment.max_episode_steps
