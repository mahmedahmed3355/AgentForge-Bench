from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionState:
    episode_id: str = "decision-v1-default"
    seed: int = 0

    step: int = 0
    logical_stage: int = 0

    current_component: str = "source"
    system_status: str = "healthy"

    observed_information: set[str] = field(default_factory=set)
    hidden_information: dict[str, Any] = field(default_factory=dict)

    belief_state: dict[str, float] = field(default_factory=dict)
    hypothesis: str | None = None
    hypothesis_confidence: float = 0.0

    selected_branch: str | None = None
    selected_strategy: str | None = None

    decision_cost: float = 0.0
    risk_level: float = 0.0
    remaining_budget: float = 100.0

    component_states: dict[str, str] = field(default_factory=dict)
    dependency_states: dict[str, str] = field(default_factory=dict)

    failure_detected: bool = False
    failure_source: str | None = None
    downstream_effect: str | None = None

    recovery_required: bool = False
    recovery_completed: bool = False

    replanning_required: bool = False
    replan_completed: bool = False

    validation_status: str = "not_validated"

    terminal: bool = False
    success: bool = False

    action_history: list[str] = field(default_factory=list)
    observation_history: list[dict[str, Any]] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    recovery_history: list[dict[str, Any]] = field(default_factory=list)

    delayed_events: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "DecisionState":
        return DecisionState(
            episode_id=self.episode_id,
            seed=self.seed,
            step=self.step,
            logical_stage=self.logical_stage,
            current_component=self.current_component,
            system_status=self.system_status,
            observed_information=set(self.observed_information),
            hidden_information=dict(self.hidden_information),
            belief_state=dict(self.belief_state),
            hypothesis=self.hypothesis,
            hypothesis_confidence=self.hypothesis_confidence,
            selected_branch=self.selected_branch,
            selected_strategy=self.selected_strategy,
            decision_cost=self.decision_cost,
            risk_level=self.risk_level,
            remaining_budget=self.remaining_budget,
            component_states=dict(self.component_states),
            dependency_states=dict(self.dependency_states),
            failure_detected=self.failure_detected,
            failure_source=self.failure_source,
            downstream_effect=self.downstream_effect,
            recovery_required=self.recovery_required,
            recovery_completed=self.recovery_completed,
            replanning_required=self.replanning_required,
            replan_completed=self.replan_completed,
            validation_status=self.validation_status,
            terminal=self.terminal,
            success=self.success,
            action_history=list(self.action_history),
            observation_history=[dict(x) for x in self.observation_history],
            decision_history=[dict(x) for x in self.decision_history],
            failure_history=[dict(x) for x in self.failure_history],
            recovery_history=[dict(x) for x in self.recovery_history],
            delayed_events=[dict(x) for x in self.delayed_events],
        )

    def record_action(self, action: str) -> None:
        self.action_history.append(action)
        self.step += 1

    def spend(self, cost: float) -> bool:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        if cost > self.remaining_budget:
            return False

        self.remaining_budget -= cost
        self.decision_cost += cost
        return True

    def add_information(self, key: str) -> bool:
        if key in self.observed_information:
            return False

        self.observed_information.add(key)
        self.logical_stage += 1
        return True

    def register_decision(
        self,
        branch: str,
        strategy: str | None = None,
    ) -> None:
        self.selected_branch = branch

        if strategy is not None:
            self.selected_strategy = strategy

        self.decision_history.append(
            {
                "step": self.step,
                "branch": branch,
                "strategy": strategy,
                "logical_stage": self.logical_stage,
            }
        )

        self.logical_stage += 1

    def register_failure(
        self,
        source: str,
        effect: str,
    ) -> None:
        self.failure_detected = True
        self.failure_source = source
        self.downstream_effect = effect
        self.system_status = "failed"
        self.recovery_required = True
        self.replanning_required = True

        self.failure_history.append(
            {
                "step": self.step,
                "source": source,
                "effect": effect,
            }
        )

        self.logical_stage += 1

    def register_recovery(self, strategy: str) -> None:
        self.recovery_history.append(
            {
                "step": self.step,
                "strategy": strategy,
            }
        )

        self.recovery_completed = True
        self.recovery_required = False
        self.logical_stage += 1

    def register_replan(self) -> None:
        self.replan_completed = True
        self.replanning_required = False
        self.logical_stage += 1

    def validate_invariants(self) -> None:
        if self.step < 0:
            raise AssertionError("step cannot be negative")

        if self.logical_stage < 0:
            raise AssertionError("logical_stage cannot be negative")

        if self.remaining_budget < 0:
            raise AssertionError("remaining_budget cannot be negative")

        if not 0.0 <= self.hypothesis_confidence <= 1.0:
            raise AssertionError("hypothesis_confidence must be in [0, 1]")

        if self.success and not self.terminal:
            raise AssertionError("success requires terminal state")

        if self.recovery_completed and self.recovery_required:
            raise AssertionError(
                "recovery cannot be completed while recovery is required"
            )

        if self.replan_completed and self.replanning_required:
            raise AssertionError(
                "replan cannot be completed while replanning is required"
            )
