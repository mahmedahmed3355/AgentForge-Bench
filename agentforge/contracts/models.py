from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TaskData:
    idx: int
    name: str
    description: str
    prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    step: int
    state: Mapping[str, Any]
    available_actions: Sequence[str] = field(default_factory=tuple)
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardResult:
    value: float
    components: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float = 0.0
    metrics: Mapping[str, float] = field(default_factory=dict)
    failures: Sequence[str] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OracleResult:
    passed: bool
    score: float = 0.0
    steps: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEvent:
    step: int
    event_type: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrace:
    task_name: str
    scenario_id: str
    seed: int
    events: list[TraceEvent] = field(default_factory=list)

    def add(
        self,
        step: int,
        event_type: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                step=step,
                event_type=event_type,
                data={} if data is None else dict(data),
            )
        )
