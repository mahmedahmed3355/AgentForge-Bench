from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Transition:
    step: int
    stage_before: int
    stage_after: int
    action: str
    reward: float
    success: bool
    terminal: bool
    observation: Any = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)

    def append(
        self,
        transition: Transition | None = None,
        *,
        step: int | None = None,
        stage_before: int | None = None,
        stage_after: int | None = None,
        action: str | None = None,
        reward: float | None = None,
        success: bool = False,
        terminal: bool = False,
        observation: Any = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        if transition is not None:
            if not isinstance(transition, Transition):
                raise TypeError("transition must be a Transition instance")
            self.transitions.append(transition)
            return

        if step is None:
            step = len(self.transitions) + 1

        if stage_before is None:
            stage_before = stage_after if stage_after is not None else 0

        if stage_after is None:
            stage_after = stage_before

        if action is None:
            raise TypeError("action is required")

        if reward is None:
            raise TypeError("reward is required")

        self.transitions.append(
            Transition(
                step=step,
                stage_before=stage_before,
                stage_after=stage_after,
                action=action,
                reward=reward,
                success=success,
                terminal=terminal,
                observation=observation,
                info=dict(info or {}),
            )
        )

    @property
    def total_reward(self) -> float:
        return sum(item.reward for item in self.transitions)

    @property
    def action_count(self) -> int:
        return len(self.transitions)

    @property
    def recovery_count(self) -> int:
        return sum(
            1
            for item in self.transitions
            if item.action == "recover"
        )

    @property
    def terminal_success(self) -> bool:
        return bool(
            self.transitions
            and self.transitions[-1].action == "final_verify"
            and self.transitions[-1].reward > 0
            and self.transitions[-1].success
        )

    def actions(self) -> list[str]:
        return [item.action for item in self.transitions]

    def observations(self) -> list[Any]:
        return [item.observation for item in self.transitions]
