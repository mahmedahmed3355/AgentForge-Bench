from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class RLAgent(ABC):
    """Canonical lifecycle contract for an RL-compatible agent.

    The agent owns interaction-level state while policy implementations own
    action-selection behavior.
    """

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None) -> None:
        """Reset agent state at the beginning of an episode."""
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Select an action from the current observation."""
        raise NotImplementedError

    def observe(
        self,
        observation: Any,
        *,
        reward: float | None = None,
        terminated: bool = False,
        truncated: bool = False,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Receive the result of an environment transition."""
        del observation, reward, terminated, truncated, info

    def finish(
        self,
        *,
        terminated: bool = False,
        truncated: bool = False,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Finalize the current episode interaction."""
        del terminated, truncated, info

    def metadata(self) -> Mapping[str, Any]:
        """Return descriptive agent metadata."""
        return {}
