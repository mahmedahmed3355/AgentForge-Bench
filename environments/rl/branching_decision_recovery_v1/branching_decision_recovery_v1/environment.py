from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import ActionKind
from .inspect_adapter import InspectAdapter, InspectionResult
from .decision_graph import (
    DecisionBranch,
    DecisionStrategy,
    FailureKind,
    get_branch_profile,
    get_strategy_profile,
    is_strategy_compatible,
)
from .observations import Observation
from .state import DecisionState


@dataclass
class BranchingDecisionRecoveryEnv:
    scenario_id: str = "decision-v1-public-001"
    seed: int = 3001

    def __post_init__(self) -> None:
        self.inspect = InspectAdapter()
        self.state: DecisionState | None = None

    scenario_config: Any | None = None

    def reset(
        self,
        *,
        scenario_id: str | None = None,
        seed: int | None = None,
    ) -> Observation:
        if scenario_id is not None:
            self.scenario_id = scenario_id

        if seed is not None:
            self.seed = seed

        component_states = {
            "source": "healthy",
            "configuration": "unknown",
            "processing": "unknown",
            "dependency": "unknown",
            "state_store": "unknown",
            "validation": "unknown",
            "output": "unknown",
        }

        dependency_states = {
            "primary_dependency": "unknown",
        }

        hidden_information = {
            "incident_family": self._incident_family(),
        }

        if self.scenario_config is not None:
            config = self.scenario_config

            # Bind the generated topology into the observable system state.
            for node in config.topology:
                component_states.setdefault(node, "unknown")

            # Bind generated dependencies into the runtime dependency state.
            for source, target in config.dependencies:
                dependency_states[f"{source}->{target}"] = "unknown"

            # The symptom is observable as a system warning signal, not as
            # ground truth about the correct branch or recovery strategy.
            hidden_information["symptom_profile"] = config.symptom_profile

            symptom_component = {
                "latency_spike": "processing",
                "validation_drift": "validation",
                "dependency_mismatch": "dependency",
                "resource_pressure": "state_store",
            }.get(config.symptom_profile)

            if symptom_component is not None:
                component_states[symptom_component] = "degraded"

        self.state = DecisionState(
            episode_id=self.scenario_id,
            seed=self.seed,
            component_states=component_states,
            dependency_states=dependency_states,
            hidden_information=hidden_information,
        )

        self._apply_scenario_topology()
        self.state.validate_invariants()
        return self._observation()

    def step(
        self,
        action: ActionKind | str,
        **kwargs: Any,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step().")

        action = ActionKind(action)
        state = self.state

        if state.terminal:
            return (
                self._observation(),
                0.0,
                True,
                {"status": "already_terminal"},
            )

        state.record_action(action.value)

        reward = 0.0
        info: dict[str, Any] = {
            "action": action.value,
            "valid": True,
        }

        if action == ActionKind.INSPECT_SYSTEM:
            result = self.inspect.inspect_system(state)
            new_information = self._record_inspection_progress(result)
            reward = 1.0 if new_information else 0.0
            info["inspection"] = self._inspection_dict(result)
            info["inspection"]["new_information"] = new_information

        elif action == ActionKind.INSPECT_COMPONENT:
            result = self.inspect.inspect_component(
                state,
                kwargs.get("component", "processing"),
            )
            new_information = self._record_inspection_progress(result)
            reward = 1.0 if new_information else 0.0
            info["inspection"] = self._inspection_dict(result)
            info["inspection"]["new_information"] = new_information

        elif action == ActionKind.INSPECT_DEPENDENCY:
            result = self.inspect.inspect_dependency(
                state,
                kwargs.get("dependency", "primary_dependency"),
            )
            new_information = self._record_inspection_progress(result)
            reward = 1.0 if new_information else 0.0
            info["inspection"] = self._inspection_dict(result)
            info["inspection"]["new_information"] = new_information

        elif action == ActionKind.INSPECT_HISTORY:
            result = self.inspect.inspect_history(state)
            new_information = self._record_inspection_progress(result)
            reward = 1.0 if new_information else 0.0
            info["inspection"] = self._inspection_dict(result)
            info["inspection"]["new_information"] = new_information

        elif action == ActionKind.PROBE_STATE:
            result = self.inspect.probe_state(state)
            new_information = self._record_inspection_progress(result)
            reward = 1.0 if new_information else 0.0
            info["inspection"] = self._inspection_dict(result)
            info["inspection"]["new_information"] = new_information

        elif action == ActionKind.QUERY_VALIDATION:
            result = self.inspect.query_validation(state)
            new_information = self._record_inspection_progress(result)
            reward = 1.0 if new_information else 0.0
            info["inspection"] = self._inspection_dict(result)
            info["inspection"]["new_information"] = new_information

        elif action == ActionKind.FORM_HYPOTHESIS:
            hypothesis = kwargs.get("hypothesis")
            confidence = float(kwargs.get("confidence", 0.0))

            if hypothesis is None:
                reward = -0.5
                info["valid"] = False
            else:
                state.hypothesis = str(hypothesis)
                state.hypothesis_confidence = max(
                    0.0,
                    min(1.0, confidence),
                )
                state.logical_stage += 1
                reward = 1.0

        elif action == ActionKind.SELECT_BRANCH:
            branch = kwargs.get("branch")

            if branch not in {"branch_a", "branch_b", "branch_c"}:
                reward = -1.0
                info["valid"] = False
            else:
                state.selected_branch = branch
                state.logical_stage += 1
                reward = 1.0

        elif action == ActionKind.SELECT_STRATEGY:
            strategy = kwargs.get("strategy")

            if strategy not in {
                "strategy_a",
                "strategy_b",
                "strategy_c",
            }:
                reward = -1.0
                info["valid"] = False
            else:
                state.selected_strategy = strategy
                state.logical_stage += 1
                reward = 1.0

        elif action == ActionKind.APPLY_ACTION:
            reward = self._apply_costly_action(kwargs)

        elif action == ActionKind.MODIFY_CONFIGURATION:
            if not state.spend(float(kwargs.get("cost", 4.0))):
                reward = -1.0
                info["valid"] = False
            else:
                state.logical_stage += 1
                reward = 0.5

        elif action == ActionKind.RUN:
            reward = self._run()

        elif action == ActionKind.VALIDATE:
            reward = self._validate()

        elif action == ActionKind.RECOVER:
            reward = self._recover(kwargs)

        elif action == ActionKind.REPLAN:
            reward = self._replan()

        elif action == ActionKind.FINAL_VERIFY:
            reward = self._final_verify()

        state.validate_invariants()

        observation = self._observation()

        return observation, reward, state.terminal, info

    def _apply_costly_action(self, kwargs: dict[str, Any]) -> float:
        assert self.state is not None

        if self.state.selected_branch is None:
            requested_cost = float(kwargs.get("cost", 6.0))

            if not self.state.spend(requested_cost):
                return -1.0

            return -1.0

        branch = DecisionBranch(self.state.selected_branch)

        if self.state.selected_strategy is not None:
            strategy = DecisionStrategy(self.state.selected_strategy)

            if not is_strategy_compatible(branch, strategy):
                self.state.system_status = "invalid_strategy"
                return -1.0

            strategy_profile = self._effective_strategy_profile(strategy)

            if strategy_profile.requires_information:
                if not self.state.observed_information:
                    return -1.0

        branch_profile = self._effective_branch_profile(branch)

        requested_cost = float(
            kwargs.get("cost", branch_profile.cost)
        )

        if not self.state.spend(requested_cost):
            return -1.0

        self.state.logical_stage += 2
        self.state.risk_level = branch_profile.risk

        if branch_profile.delayed_consequence:
            self.state.system_status = "locally_valid"

            if branch_profile.failure_kind is not None:
                event = {
                    "trigger_step": (
                        self.state.step + branch_profile.consequence_delay
                    ),
                    "type": branch_profile.failure_kind.value,
                    "source": branch.value,
                    "effect": branch_profile.downstream_effect,
                }

                self.state.delayed_events.append(event)
        else:
            self.state.system_status = "stable"
            self.state.component_states["processing"] = "healthy"

        self._process_delayed_events()

        return 1.0

    def _effective_branch_profile(self, branch: DecisionBranch):
        """Return the scenario-bound branch profile.

        Public scenarios preserve the canonical DecisionGraph profiles.
        Hidden/generated scenarios override only the parameters supplied by
        ScenarioConfig while retaining the branch semantics.
        """
        profile = get_branch_profile(branch)

        if self.scenario_config is None:
            return profile

        from dataclasses import replace

        index = {
            DecisionBranch.BRANCH_A: 0,
            DecisionBranch.BRANCH_B: 1,
            DecisionBranch.BRANCH_C: 2,
        }[branch]

        config = self.scenario_config

        return replace(
            profile,
            cost=float(config.branch_costs[index]),
            risk=float(config.branch_risks[index]),
            consequence_delay=(
                int(config.consequence_delay)
                if profile.delayed_consequence
                else 0
            ),
        )

    def _effective_strategy_profile(self, strategy: DecisionStrategy):
        """Return the scenario-bound strategy profile."""
        profile = get_strategy_profile(strategy)

        if self.scenario_config is None:
            return profile

        from dataclasses import replace

        index = {
            DecisionStrategy.STRATEGY_A: 0,
            DecisionStrategy.STRATEGY_B: 1,
            DecisionStrategy.STRATEGY_C: 2,
        }[strategy]

        config = self.scenario_config

        return replace(
            profile,
            cost=float(config.strategy_costs[index]),
            risk=float(config.strategy_risks[index]),
        )

    def _apply_scenario_topology(self) -> None:
        """Apply generated topology/dependencies without exposing hidden truth."""
        assert self.state is not None

        config = self.scenario_config
        if config is None:
            return

        for source, target in config.dependencies:
            key = f"{source}->{target}"
            self.state.dependency_states.setdefault(key, "unknown")

    def _run(self) -> float:
        assert self.state is not None

        self.state.logical_stage += 2
        self._process_delayed_events()

        if self.state.failure_detected:
            return 0.0

        return 1.0

    def _validate(self) -> float:
        assert self.state is not None

        self.state.logical_stage += 1

        # Validation is deliberately two-phase:
        # 1. A first healthy validation establishes local validity.
        # 2. A subsequent healthy validation after execution establishes
        #    global validity.
        #
        # Failure states can never be promoted to global validation.
        if self.state.failure_detected:
            self.state.validation_status = "local_validated"
            return 0.0

        if self.state.validation_status == "local_validated":
            self.state.validation_status = "globally_validated"
            return 1.0

        self.state.validation_status = "local_validated"
        return 0.75

    def _recover(self, kwargs: dict[str, Any]) -> float:
        assert self.state is not None

        if not self.state.recovery_required:
            return 0.0

        strategy = str(kwargs.get("strategy", "reconfigure"))
        cost = float(kwargs.get("cost", 8.0))

        if not self.state.spend(cost):
            return -1.0

        self.state.register_recovery(strategy)

        # Recovery changes the current causal system state, but must not erase
        # the historical failure recorded in failure_history. The environment
        # therefore clears only the active failure condition and restores the
        # affected runtime state to a recoverable baseline.
        self.state.failure_detected = False
        self.state.failure_source = None
        self.state.downstream_effect = None
        self.state.validation_status = "not_validated"

        self.state.system_status = "recovering"

        for component, component_state in list(self.state.component_states.items()):
            if component_state in {"failed", "invalid", "blocked", "degraded"}:
                self.state.component_states[component] = "recovering"

        for dependency, dependency_state in list(self.state.dependency_states.items()):
            if dependency_state in {"failed", "invalid", "blocked", "degraded"}:
                self.state.dependency_states[dependency] = "recovering"

        return 1.5

    def _replan(self) -> float:
        assert self.state is not None

        if not self.state.replanning_required:
            return 0.0

        self.state.register_replan()

        # Replanning is a causal transition, not a terminal success signal.
        # It clears the active recovery/replan requirement while preserving
        # historical failures and forcing a fresh execution/validation cycle.
        self.state.failure_detected = False
        self.state.failure_source = None
        self.state.downstream_effect = None
        self.state.validation_status = "not_validated"

        self.state.system_status = "replanned"

        for component, component_state in list(self.state.component_states.items()):
            if component_state in {"recovering", "failed", "invalid", "blocked", "degraded"}:
                self.state.component_states[component] = "ready"

        for dependency, dependency_state in list(self.state.dependency_states.items()):
            if dependency_state in {"recovering", "failed", "invalid", "blocked", "degraded"}:
                self.state.dependency_states[dependency] = "ready"

        return 1.5

    def _final_verify(self) -> float:
        assert self.state is not None

        if (
            self.state.recovery_required
            or self.state.replanning_required
            or self.state.failure_detected
        ):
            self.state.terminal = True
            self.state.success = False
            self.state.system_status = "unrecoverable_failure"
            return -5.0

        if (
            self.state.selected_branch is None
            or self.state.selected_strategy is None
            or self.state.validation_status != "globally_validated"
        ):
            return 0.0

        self.state.terminal = True
        self.state.success = True
        self.state.system_status = "success"

        return 10.0

    def _process_delayed_events(self) -> None:
        assert self.state is not None

        pending = []

        for event in self.state.delayed_events:
            if event["trigger_step"] <= self.state.step:
                event_type = event["type"]

                if event_type == FailureKind.DOWNSTREAM.value:
                    self.state.register_failure(
                        source="downstream_dependency",
                        effect=event["effect"],
                    )

                elif event_type == FailureKind.CASCADE.value:
                    self.state.register_failure(
                        source="dependency_cascade",
                        effect=event["effect"],
                    )

                elif event_type == FailureKind.LOCAL.value:
                    self.state.register_failure(
                        source="local_component",
                        effect=event["effect"],
                    )

                elif event_type == FailureKind.DEPENDENCY.value:
                    self.state.register_failure(
                        source="dependency",
                        effect=event["effect"],
                    )

                else:
                    pending.append(event)
            else:
                pending.append(event)

        self.state.delayed_events = pending

    def _observation(self) -> Observation:
        assert self.state is not None
        observation = self.inspect.observation(self.state)
        self.state.observation_history.append(observation.to_dict())
        return observation

    def _record_inspection_progress(self, result: InspectionResult) -> bool:
        assert self.state is not None
        return self.state.add_information(result.information_key)

    def _inspection_dict(self, result: InspectionResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "information_key": result.information_key,
            "value": result.value,
            "cost": result.cost,
            "new_information": result.new_information,
        }

    def _incident_family(self) -> str:
        families = {
            "decision-v1-public-001": "dependency",
            "decision-v1-public-002": "configuration",
            "decision-v1-public-003": "cascade",
        }
        return families.get(self.scenario_id, "dependency")
