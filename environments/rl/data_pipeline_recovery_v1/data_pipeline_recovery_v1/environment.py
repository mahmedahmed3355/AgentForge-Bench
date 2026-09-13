from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import ActionKind
from .causal import (
    CausalPipeline,
    PipelineComponent,
)
from .decision_graph import (
    IncidentKind,
    RecoveryMode,
    RecoveryStrategy,
    get_branch,
    get_strategy_profile,
)
from .observation import PipelineObservation
from .state import PipelineState


MAX_ACTIONS = 40
MIN_FINAL_STAGE = 14


@dataclass(frozen=True)
class StepResult:
    observation: PipelineObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class DataPipelineRecoveryEnv:

    def __init__(
        self,
        episode_id: str = "pipeline-v1-episode",
        incident: IncidentKind = IncidentKind.SCHEMA,
    ) -> None:
        self.episode_id = episode_id
        self.incident = incident
        self.state = PipelineState(
            episode_id=episode_id,
            incident=incident,
        )
        self.pipeline = CausalPipeline()
        self._pipeline_inspected = False
        self._inspected_components: set[str] = set()

    def reset(self) -> PipelineObservation:
        self.state = PipelineState(
            episode_id=self.episode_id,
            incident=self.incident,
        )
        self.pipeline.reset()
        self._pipeline_inspected = False
        self._inspected_components.clear()
        self._sync_pipeline_state()
        return PipelineObservation.from_state(self.state)

    def _sync_pipeline_state(self) -> None:
        snapshot = self.pipeline.snapshot()

        self.state.component_validity = snapshot

        self.state.schema_valid = snapshot["schema_validation"]
        self.state.cleaning_valid = snapshot["cleaning"]
        self.state.transformation_valid = snapshot["transformation"]
        self.state.aggregation_valid = snapshot["aggregation"]
        self.state.output_valid = snapshot["output_validation"]

        self.state.blocked_components = {
            status.component.value
            for status in self.pipeline.statuses()
            if status.blocked
        }

    def _apply_component_validity(
        self,
        component: PipelineComponent,
    ) -> None:
        self.pipeline.set_valid(component, True)
        self.pipeline.invalidate_downstream(component)
        self._sync_pipeline_state()

    def _advance(self, amount: int = 1) -> None:
        self.state.logical_stage += amount

    def _result(
        self,
        reward: float,
        *,
        info: dict[str, Any] | None = None,
    ) -> StepResult:

        self.state.action_count += 1

        truncated = (
            self.state.action_count >= MAX_ACTIONS
            and not self.state.terminal
        )

        if truncated:
            self.state.terminal = True

        return StepResult(
            observation=PipelineObservation.from_state(self.state),
            reward=reward,
            terminated=self.state.terminal,
            truncated=truncated,
            info=info or {},
        )

    def _invalid(
        self,
        reason: str,
        reward: float = -1.0,
    ) -> StepResult:

        return self._result(
            reward,
            info={
                "valid": False,
                "reason": reason,
            },
        )

    def _select_strategy(
        self,
        strategy_value: str,
    ) -> RecoveryStrategy | None:

        try:
            strategy = RecoveryStrategy(strategy_value)
        except ValueError:
            return None

        try:
            profile = get_strategy_profile(
                self.incident,
                strategy,
            )
        except ValueError:
            return None

        self.state.selected_strategy = strategy
        self.state.recovery_mode = profile.mode

        return strategy

    def _apply_local_repair(self) -> None:

        strategy = self.state.selected_strategy

        if strategy == RecoveryStrategy.REPAIR_UPSTREAM:
            self._apply_component_validity(
                PipelineComponent.SCHEMA
            )

        elif strategy == RecoveryStrategy.REPAIR_CONSUMER:
            self._apply_component_validity(
                PipelineComponent.SCHEMA
            )
            self.state.downstream_inconsistency = True
            self.state.replanning_required = True

        elif strategy == RecoveryStrategy.PATCH_TRANSFORM:
            self._apply_component_validity(
                PipelineComponent.TRANSFORMATION
            )
            self.state.downstream_inconsistency = True
            self.state.replanning_required = True

        elif strategy == RecoveryStrategy.ROLLBACK_RECOMPUTE:
            self._apply_component_validity(
                PipelineComponent.TRANSFORMATION
            )
            self.state.downstream_inconsistency = True

        elif strategy == RecoveryStrategy.FILTER_INVALID:
            self._apply_component_validity(
                PipelineComponent.CLEANING
            )
            self.state.downstream_inconsistency = True
            self.state.replanning_required = True

        elif strategy == RecoveryStrategy.RECONSTRUCT_INVALID:
            self._apply_component_validity(
                PipelineComponent.CLEANING
            )
            self.state.downstream_inconsistency = True

        elif strategy == RecoveryStrategy.OPTIMIZE_EXECUTION:
            self.state.recovery_required = True
            self.state.replanning_required = True

        elif strategy == RecoveryStrategy.RETRY_WITH_RECOVERY:
            self.state.recovery_required = True

    def _complete_recovery(self) -> None:

        strategy = self.state.selected_strategy

        if strategy in (
            RecoveryStrategy.REPAIR_UPSTREAM,
            RecoveryStrategy.REPAIR_CONSUMER,
        ):
            self.state.schema_valid = True

        elif strategy in (
            RecoveryStrategy.PATCH_TRANSFORM,
            RecoveryStrategy.ROLLBACK_RECOMPUTE,
        ):
            self.state.transformation_valid = True
            self.state.aggregation_valid = True

        elif strategy in (
            RecoveryStrategy.FILTER_INVALID,
            RecoveryStrategy.RECONSTRUCT_INVALID,
        ):
            self.state.cleaning_valid = True
            self.state.transformation_valid = True
            self.state.aggregation_valid = True

        elif strategy in (
            RecoveryStrategy.OPTIMIZE_EXECUTION,
            RecoveryStrategy.RETRY_WITH_RECOVERY,
        ):
            self.state.schema_valid = True
            self.state.cleaning_valid = True
            self.state.transformation_valid = True
            self.state.aggregation_valid = True

        self.state.recovery_completed = True
        self.state.recovery_required = False
        self.state.downstream_inconsistency = False

        # Recovery establishes the affected stage and its required
        # downstream stages, but final validation still requires
        # explicit execution and output validation.
        self.pipeline.set_valid(
            PipelineComponent.SCHEMA,
            self.state.schema_valid,
        )
        self.pipeline.set_valid(
            PipelineComponent.CLEANING,
            self.state.cleaning_valid,
        )
        self.pipeline.set_valid(
            PipelineComponent.TRANSFORMATION,
            self.state.transformation_valid,
        )
        self.pipeline.set_valid(
            PipelineComponent.AGGREGATION,
            self.state.aggregation_valid,
        )

        if strategy in (
            RecoveryStrategy.OPTIMIZE_EXECUTION,
            RecoveryStrategy.RETRY_WITH_RECOVERY,
        ):
            self.pipeline.set_valid(
                PipelineComponent.OUTPUT_VALIDATION,
                True,
            )
            self.pipeline.set_valid(
                PipelineComponent.OUTPUT,
                True,
            )

        self._sync_pipeline_state()

    def step(
        self,
        action: ActionKind,
        **kwargs: Any,
    ) -> StepResult:

        if self.state.terminal:
            return self._invalid(
                "episode_already_terminal",
                reward=0.0,
            )

        if action == ActionKind.INSPECT_PIPELINE:

            first_inspection = not self._pipeline_inspected
            self._pipeline_inspected = True

            self.state.incident_observed = True
            self.state.diagnosis_started = True

            if first_inspection:
                self._advance(1)

            return self._result(
                1.0 if first_inspection else 0.0,
                info={
                    "phase": "inspection",
                    "action": action.value,
                    "new_information": first_inspection,
                },
            )

        if action == ActionKind.INSPECT_COMPONENT:

            component = str(
                kwargs.get("component", "")
            )

            if not component:
                return self._invalid(
                    "component_required",
                    -0.5,
                )

            first_component_inspection = (
                component not in self._inspected_components
            )
            self._inspected_components.add(component)
            self.state.inspected_components.add(component)
            if first_component_inspection:
                self._advance(1)

            return self._result(
                0.75 if first_component_inspection else 0.0,
                info={
                    "phase": "component_inspection",
                    "component": component,
                },
            )

        if action == ActionKind.ANALYZE_FAILURE:

            if not self.state.diagnosis_started:
                return self._invalid(
                    "diagnosis_not_started",
                    -0.5,
                )

            # Backward-compatible diagnosis semantics.
            # The legacy execution contract allowed direct analysis
            # immediately after pipeline inspection.
            self.state.root_cause_identified = True
            self._advance(2)

            return self._result(
                1.5,
                info={
                    "phase": "root_cause_analysis",
                },
            )

        if action == ActionKind.SELECT_BRANCH:

            if not self.state.root_cause_identified:
                return self._invalid(
                    "root_cause_required",
                    -1.0,
                )

            requested = str(
                kwargs.get("branch", "")
            )

            expected = get_branch(self.incident)

            if requested != expected.name:
                return self._invalid(
                    "incorrect_branch",
                    -1.5,
                )

            self.state.selected_branch = expected.name
            self._advance(1)

            return self._result(
                1.0,
                info={
                    "phase": "branch_selection",
                    "branch": expected.name,
                },
            )

        if action == ActionKind.SELECT_STRATEGY:

            if not self.state.branch_selected:
                return self._invalid(
                    "branch_required",
                    -1.0,
                )

            requested = str(
                kwargs.get("strategy", "")
            )

            strategy = self._select_strategy(requested)

            if strategy is None:
                return self._invalid(
                    "invalid_strategy_for_branch",
                    -1.5,
                )

            self._advance(1)

            return self._result(
                1.0,
                info={
                    "phase": "strategy_selection",
                    "strategy": strategy.value,
                    "mode": self.state.recovery_mode.value,
                },
            )

        if action == ActionKind.MODIFY_COMPONENT:

            if not self.state.strategy_selected:
                return self._invalid(
                    "strategy_required",
                    -1.0,
                )

            component = str(
                kwargs.get("component", "")
            )

            if not component:
                component = self.state.selected_strategy.value

            self._apply_local_repair()
            self._advance(1)

            return self._result(
                1.25,
                info={
                    "phase": "local_repair",
                    "component": component,
                    "strategy": self.state.selected_strategy.value,
                },
            )

        if action == ActionKind.MODIFY_CONFIG:

            if not self.state.strategy_selected:
                return self._invalid(
                    "strategy_required",
                    -0.75,
                )

            self._advance(1)

            return self._result(
                1.0,
                info={
                    "phase": "configuration_change",
                },
            )

        if action == ActionKind.RUN_PIPELINE:
            if self.state.selected_strategy is None:
                return self._invalid(
                    "strategy_required",
                    reward=-1.0,
                )

            self.state.logical_stage += 2

            invalid_components = []
            blocked_components = []

            for component in (
                PipelineComponent.SCHEMA,
                PipelineComponent.CLEANING,
                PipelineComponent.TRANSFORMATION,
                PipelineComponent.AGGREGATION,
                PipelineComponent.OUTPUT_VALIDATION,
                PipelineComponent.OUTPUT,
            ):
                dependencies_valid = all(
                    self.pipeline.is_valid(parent)
                    for parent in self.pipeline.dependencies(component)
                )

                if not dependencies_valid:
                    invalid_components.append(component.value)
                    if self.pipeline.is_blocked(component):
                        blocked_components.append(component.value)
                    break

                self.pipeline.set_valid(component, True)

            self._sync_pipeline_state()

            run = self.pipeline.run()
            self.state.pipeline_run_completed = run.completed

            if (
                invalid_components
                or not run.globally_valid
                or self.state.downstream_inconsistency
            ):
                self.state.failure_detected = True
                self.state.recovery_required = True

                remaining_invalid = tuple(
                    component
                    for component in run.invalid_components
                    if component not in invalid_components
                )

                return self._result(
                    reward=-2.0,
                    info={
                        "phase": "execution",
                        "status": "downstream_failure",
                        "result": "downstream_failure",
                        "invalid_components": tuple(
                            invalid_components
                        ) + remaining_invalid,
                        "blocked_components": tuple(
                            blocked_components
                        ),
                    },
                )

            return self._result(
                reward=2.0,
                info={
                    "phase": "execution",
                    "status": "completed",
                    "result": "completed",
                    "invalid_components": (),
                    "blocked_components": (),
                },
            )

        if action == ActionKind.CHECK_SCHEMA:

            self._advance(1)

            if self.state.schema_valid:
                return self._result(
                    1.0,
                    info={
                        "phase": "schema_validation",
                        "valid": True,
                    },
                )

            self.state.failure_detected = True

            return self._result(
                -1.0,
                info={
                    "phase": "schema_validation",
                    "valid": False,
                },
            )

        if action == ActionKind.CHECK_OUTPUT:

            self._advance(1)

            valid = (
                self.state.all_components_valid
                and self.state.pipeline_run_completed
                and not self.state.downstream_inconsistency
            )

            if valid:

                self.state.output_valid = True

                return self._result(
                    2.0,
                    info={
                        "phase": "output_validation",
                        "valid": True,
                    },
                )

            self.state.failure_detected = True

            return self._result(
                -1.5,
                info={
                    "phase": "output_validation",
                    "valid": False,
                },
            )

        if action == ActionKind.RECOVER:

            if not self.state.recovery_required:
                return self._invalid(
                    "recovery_not_required",
                    -0.5,
                )

            self._complete_recovery()
            self._advance(2)

            return self._result(
                2.0,
                info={
                    "phase": "recovery",
                    "strategy": self.state.selected_strategy.value,
                },
            )

        if action == ActionKind.REPLAN:

            if not self.state.replanning_required:
                return self._invalid(
                    "replanning_not_required",
                    -0.5,
                )

            self.state.replanning_required = False
            self.state.replan_completed = True
            self._advance(2)

            return self._result(
                1.5,
                info={
                    "phase": "replanning",
                    "status": "plan_updated",
                },
            )

        if action == ActionKind.FINAL_VERIFY:

            self._advance(1)

            valid = (
                self.state.logical_stage >= MIN_FINAL_STAGE
                and self.state.root_cause_identified
                and self.state.branch_selected
                and self.state.strategy_selected
                and self.state.pipeline_run_completed
                and self.state.all_components_valid
                and self.state.output_valid
                and not self.state.recovery_required
                and not self.state.replanning_required
                and not self.state.downstream_inconsistency
            )

            self.state.global_validation_completed = valid
            self.state.strategy_validated = valid
            self.state.success = valid

            if valid:
                self.state.terminal = True
                reward = 10.0
            else:
                self.state.terminal = False
                reward = -5.0

            return self._result(
                reward,
                info={
                    "phase": "final_verification",
                    "success": valid,
                    "global_success": valid,
                    "terminal": self.state.terminal,
                    "verification_failed": not valid,
                },
            )

        return self._invalid(
            f"unsupported_action:{action}",
            -1.0,
        )
