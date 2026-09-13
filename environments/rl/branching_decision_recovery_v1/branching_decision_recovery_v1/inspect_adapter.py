from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InspectionResult:
    kind: str
    target: str
    value: str
    cost: float
    useful: bool
    details: dict[str, Any]

    @property
    def information_value(self) -> str:
        return self.value

    @property
    def information_key(self) -> str:
        if self.kind == "system":
            return "system_status"
        if self.kind == "component":
            return f"component:{self.target}"
        if self.kind == "dependency":
            return f"dependency:{self.target}"
        if self.kind == "history":
            return "history"
        if self.kind == "probe":
            return "probe_state"
        if self.kind == "validation":
            return "validation"
        return self.kind

    @property
    def new_information(self) -> bool:
        return bool(self.useful)

    @property
    def query(self) -> str:
        query_map = {
            "system": "inspect_system",
            "component": "inspect_component",
            "dependency": "inspect_dependency",
            "history": "inspect_history",
            "metrics": "inspect_metrics",
            "validation": "query_validation",
            "state": "probe_state",
        }
        return query_map.get(self.kind, self.kind)



class InspectAdapter:
    def __init__(self, state: Any | None = None):
        self.state = state

    def _resolve_state(self, state: Any | None = None) -> Any:
        resolved = state if state is not None else self.state

        if resolved is None:
            raise ValueError("InspectAdapter requires a state.")

        return resolved

    def inspect_system(
        self,
        state: Any | None = None,
    ) -> InspectionResult:
        return inspect_system(self._resolve_state(state))

    def observation(
        self,
        state: Any | None = None,
    ):
        resolved = self._resolve_state(state)

        from .observations import Observation, validate_observation_contract

        component_states = getattr(
            resolved,
            "component_states",
            {},
        )

        dependency_states = getattr(
            resolved,
            "dependency_states",
            {},
        )

        visible_components = tuple(
            str(name)
            for name in component_states.keys()
        ) if isinstance(component_states, dict) else ()

        warnings = tuple(
            str(item)
            for item in getattr(
                resolved,
                "warnings",
                [],
            )
        )

        observed_information = tuple(
            str(item)
            for item in getattr(
                resolved,
                "observed_information",
                [],
            )
        )

        local_metrics = getattr(
            resolved,
            "local_metrics",
            {},
        )

        if not isinstance(local_metrics, dict):
            local_metrics = {}

        if not local_metrics and isinstance(dependency_states, dict):
            local_metrics = {
                "visible_dependency_count": len(
                    dependency_states
                )
            }

        observation = Observation(
            step=int(
                getattr(
                    resolved,
                    "step",
                    0,
                )
            ),
            logical_stage=int(
                getattr(
                    resolved,
                    "logical_stage",
                    0,
                )
            ),
            system_status=str(
                getattr(
                    resolved,
                    "system_status",
                    "unknown",
                )
            ),
            current_component=getattr(
                resolved,
                "current_component",
                None,
            ),
            visible_components=visible_components,
            available_actions=tuple(
                getattr(
                    resolved,
                    "available_actions",
                    [],
                )
            ) or tuple(
                action.value
                for action in __import__(
                    "branching_decision_recovery_v1.actions",
                    fromlist=["ActionKind"],
                ).ActionKind
            ),
            observed_information=observed_information,
            warnings=warnings,
            local_metrics=dict(local_metrics),
            hypothesis=getattr(
                resolved,
                "hypothesis",
                None,
            ),
            hypothesis_confidence=float(
                getattr(
                    resolved,
                    "hypothesis_confidence",
                    0.0,
                )
            ),
            selected_branch=getattr(
                resolved,
                "selected_branch",
                None,
            ),
            selected_strategy=getattr(
                resolved,
                "selected_strategy",
                None,
            ),
            failure_detected=bool(
                getattr(
                    resolved,
                    "failure_detected",
                    False,
                )
            ),
            recovery_required=bool(
                getattr(
                    resolved,
                    "recovery_required",
                    False,
                )
            ),
            replanning_required=bool(
                getattr(
                    resolved,
                    "replanning_required",
                    False,
                )
            ),
            terminal=bool(
                getattr(
                    resolved,
                    "terminal",
                    False,
                )
            ),
            success=bool(
                getattr(
                    resolved,
                    "success",
                    False,
                )
            ),
        )

        validate_observation_contract(observation)

        return observation

    def inspect_component(
        self,
        state: Any | None = None,
        component: str | None = None,
    ) -> InspectionResult:
        if component is None and isinstance(state, str):
            component = state
            state = None

        if component is None:
            raise ValueError("inspect_component requires component.")

        return inspect_component(
            self._resolve_state(state),
            component,
        )

    def inspect_dependency(
        self,
        state: Any | None = None,
        dependency: str | None = None,
    ) -> InspectionResult:
        if dependency is None and isinstance(state, str):
            dependency = state
            state = None

        if dependency is None:
            raise ValueError("inspect_dependency requires dependency.")

        return inspect_dependency(
            self._resolve_state(state),
            dependency,
        )

    def inspect_history(
        self,
        state: Any | None = None,
    ) -> InspectionResult:
        return inspect_history(self._resolve_state(state))

    def probe_state(
        self,
        state: Any | None = None,
    ) -> InspectionResult:
        return probe_state(self._resolve_state(state))

    def query_validation(
        self,
        state: Any | None = None,
    ) -> InspectionResult:
        return query_validation(self._resolve_state(state))


def _visible_component_state(
    state: Any,
    component: str,
) -> str:
    components = getattr(state, "component_states", {})

    if isinstance(components, dict):
        return str(components.get(component, "unknown"))

    return "unknown"


def inspect_system(state: Any) -> InspectionResult:
    status = str(
        getattr(state, "system_status", "unknown")
    )

    return InspectionResult(
        kind="system",
        target="system",
        value=status,
        cost=1.0,
        useful=True,
        details={
            "system_status": status,
            "step": int(getattr(state, "step", 0)),
            "logical_stage": int(
                getattr(state, "logical_stage", 0)
            ),
        },
    )


def inspect_component(
    state: Any,
    component: str,
) -> InspectionResult:
    value = _visible_component_state(
        state,
        component,
    )

    return InspectionResult(
        kind="component",
        target=component,
        value=value,
        cost=1.0,
        useful=True,
        details={
            "component": component,
            "state": value,
        },
    )


def inspect_dependency(
    state: Any,
    dependency: str,
) -> InspectionResult:
    dependencies = getattr(
        state,
        "dependency_states",
        {},
    )

    value = "unknown"

    if isinstance(dependencies, dict):
        value = str(
            dependencies.get(
                dependency,
                "unknown",
            )
        )

    return InspectionResult(
        kind="dependency",
        target=dependency,
        value=value,
        cost=1.0,
        useful=True,
        details={
            "dependency": dependency,
            "state": value,
        },
    )


def inspect_history(state: Any) -> InspectionResult:
    history = getattr(
        state,
        "history",
        [],
    )

    length = (
        len(history)
        if hasattr(history, "__len__")
        else 0
    )

    return InspectionResult(
        kind="history",
        target="history",
        value=f"history_length={length}",
        cost=1.0,
        useful=length > 0,
        details={
            "history_length": length,
        },
    )


def probe_state(state: Any) -> InspectionResult:
    status = str(
        getattr(state, "system_status", "unknown")
    )

    failure = bool(
        getattr(
            state,
            "failure_detected",
            False,
        )
    )

    return InspectionResult(
        kind="probe",
        target="state",
        value=(
            f"status={status};"
            f"failure_detected={failure}"
        ),
        cost=2.0,
        useful=True,
        details={
            "system_status": status,
            "failure_detected": failure,
        },
    )


def query_validation(state: Any) -> InspectionResult:
    validation = str(
        getattr(
            state,
            "validation_status",
            "unknown",
        )
    )

    return InspectionResult(
        kind="validation",
        target="validation",
        value=validation,
        cost=2.0,
        useful=True,
        details={
            "validation_status": validation,
        },
    )
