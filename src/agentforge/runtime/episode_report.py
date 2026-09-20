from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .episode_runner import EpisodeResult


@dataclass(frozen=True)
class EpisodeReport:
    step_count: int
    total_reward: float
    terminated: bool
    truncated: bool
    metadata: dict[str, Any]


def build_episode_report(
    result: EpisodeResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> EpisodeReport:
    return EpisodeReport(
        step_count=result.step_count,
        total_reward=result.total_reward,
        terminated=result.terminated,
        truncated=result.truncated,
        metadata=dict(metadata or {}),
    )
