from __future__ import annotations
from math import isfinite

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecordedStep:
    observation: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]




from .trajectory import TrajectoryStep


class TrajectoryRecorder:
    def __init__(self) -> None:
        self._steps: list[RecordedStep] = []
        self._closed = False

    def record(
        self,
        *,
        observation: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> RecordedStep:
        if self._closed:
            raise RuntimeError("trajectory recorder is closed")

        if terminated and truncated:
            raise ValueError(
                "terminated and truncated cannot both be true"
            )

        step = RecordedStep(
            observation=observation,
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )

        self._steps.append(step)
        return step

    def steps(self) -> tuple[RecordedStep, ...]:
        return tuple(self._steps)

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed
