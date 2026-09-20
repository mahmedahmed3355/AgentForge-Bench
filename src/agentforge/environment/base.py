from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .state import EnvironmentState
from .transition import Transition


class NativeEnvironment(ABC):
    """Canonical AgentForge native environment lifecycle."""

    def __init__(self) -> None:
        self._state: EnvironmentState | None = None

    @property
    def state(self) -> EnvironmentState | None:
        return self._state

    @property
    def episode_id(self) -> str | None:
        if self._state is None:
            return None
        return self._state.episode_id

    @property
    def terminated(self) -> bool:
        return bool(self._state and self._state.terminated)

    @property
    def truncated(self) -> bool:
        return bool(self._state and self._state.truncated)

    @property
    def done(self) -> bool:
        return bool(self._state and self._state.done)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        episode_id = self._create_episode_id(seed)

        self._state = EnvironmentState(
            episode_id=episode_id,
            step_count=0,
            terminated=False,
            truncated=False,
            payload={},
        )

        return self._reset_impl(seed=seed, options=options)

    def step(self, action: Any) -> Transition:
        if self._state is None:
            raise RuntimeError(
                "environment must be reset before step()"
            )

        if self._state.done:
            raise RuntimeError(
                "cannot call step() after terminated/truncated; "
                "call reset() first"
            )

        transition = self._step_impl(action)

        if not isinstance(transition, Transition):
            raise TypeError(
                "_step_impl() must return Transition"
            )

        self._state.step_count += 1

        if transition.terminated:
            self._state.mark_terminated()
        elif transition.truncated:
            self._state.mark_truncated()

        return transition

    def close(self) -> None:
        self._close_impl()
        self._state = None

    def _create_episode_id(self, seed: int | None) -> str:
        if seed is None:
            return f"episode-{id(self)}-{self._next_episode_number()}"
        return f"episode-{seed}-{self._next_episode_number()}"

    def _next_episode_number(self) -> int:
        current = getattr(self, "_episode_counter", 0) + 1
        self._episode_counter = current
        return current

    @abstractmethod
    def _reset_impl(
        self,
        *,
        seed: int | None,
        options: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def _step_impl(self, action: Any) -> Transition:
        raise NotImplementedError

    def _close_impl(self) -> None:
        return None
