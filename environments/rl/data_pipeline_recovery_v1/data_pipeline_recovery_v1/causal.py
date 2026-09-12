from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PipelineComponent(str, Enum):
    INPUT = "input"
    SCHEMA = "schema_validation"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    OUTPUT_VALIDATION = "output_validation"
    OUTPUT = "output"


PIPELINE_ORDER = (
    PipelineComponent.INPUT,
    PipelineComponent.SCHEMA,
    PipelineComponent.CLEANING,
    PipelineComponent.TRANSFORMATION,
    PipelineComponent.AGGREGATION,
    PipelineComponent.OUTPUT_VALIDATION,
    PipelineComponent.OUTPUT,
)


DEPENDENCIES: dict[PipelineComponent, tuple[PipelineComponent, ...]] = {
    PipelineComponent.INPUT: (),
    PipelineComponent.SCHEMA: (
        PipelineComponent.INPUT,
    ),
    PipelineComponent.CLEANING: (
        PipelineComponent.SCHEMA,
    ),
    PipelineComponent.TRANSFORMATION: (
        PipelineComponent.CLEANING,
    ),
    PipelineComponent.AGGREGATION: (
        PipelineComponent.TRANSFORMATION,
    ),
    PipelineComponent.OUTPUT_VALIDATION: (
        PipelineComponent.AGGREGATION,
    ),
    PipelineComponent.OUTPUT: (
        PipelineComponent.OUTPUT_VALIDATION,
    ),
}


@dataclass(frozen=True)
class ComponentStatus:
    component: PipelineComponent
    valid: bool
    blocked: bool


@dataclass(frozen=True)
class PipelineRun:
    completed: bool
    globally_valid: bool
    invalid_components: tuple[str, ...]
    blocked_components: tuple[str, ...]


class CausalPipeline:
    """
    Deterministic dependency graph for the public execution model.

    This model describes causal mechanics, not hidden evaluation truth.
    Scenario-specific hidden conditions belong outside the agent package.
    """

    def __init__(self) -> None:
        self._valid: dict[PipelineComponent, bool] = {
            component: False
            for component in PIPELINE_ORDER
        }

        # The raw input is available at episode start.
        self._valid[PipelineComponent.INPUT] = True

    def reset(self) -> None:
        self._valid = {
            component: False
            for component in PIPELINE_ORDER
        }
        self._valid[PipelineComponent.INPUT] = True

    def set_valid(
        self,
        component: PipelineComponent,
        valid: bool = True,
    ) -> None:
        self._valid[component] = valid

    def is_valid(
        self,
        component: PipelineComponent,
    ) -> bool:
        return self._valid[component]

    def dependencies(
        self,
        component: PipelineComponent,
    ) -> tuple[PipelineComponent, ...]:
        return DEPENDENCIES[component]

    def is_blocked(
        self,
        component: PipelineComponent,
    ) -> bool:
        return any(
            not self._valid[parent]
            for parent in DEPENDENCIES[component]
        )

    def invalidate_downstream(
        self,
        component: PipelineComponent,
    ) -> None:
        """
        Propagate invalidity to every transitive downstream component.
        """
        started = False

        for current in PIPELINE_ORDER:
            if current == component:
                started = True
                continue

            if not started:
                continue

            if component in DEPENDENCIES[current] or any(
                not self._valid[parent]
                for parent in DEPENDENCIES[current]
            ):
                self._valid[current] = False

    def run(self) -> PipelineRun:
        invalid: list[str] = []
        blocked: list[str] = []

        for component in PIPELINE_ORDER:
            if component == PipelineComponent.INPUT:
                continue

            if not self._valid[component]:
                invalid.append(component.value)

            if self.is_blocked(component):
                blocked.append(component.value)

        globally_valid = (
            len(invalid) == 0
            and len(blocked) == 0
        )

        return PipelineRun(
            completed=True,
            globally_valid=globally_valid,
            invalid_components=tuple(invalid),
            blocked_components=tuple(blocked),
        )

    def snapshot(self) -> dict[str, bool]:
        return {
            component.value: self._valid[component]
            for component in PIPELINE_ORDER
        }

    def statuses(self) -> tuple[ComponentStatus, ...]:
        return tuple(
            ComponentStatus(
                component=component,
                valid=self._valid[component],
                blocked=self.is_blocked(component),
            )
            for component in PIPELINE_ORDER
        )
