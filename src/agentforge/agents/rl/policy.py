from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class Policy(ABC):
    """Canonical action-selection contract for RL policies."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return an action for the supplied observation."""
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None) -> None:
        """Reset policy state before an episode."""
        del seed, options

    def metadata(self) -> Mapping[str, Any]:
        """Return descriptive policy metadata."""
        return {}
