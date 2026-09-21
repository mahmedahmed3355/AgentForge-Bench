"""Canonical reward contracts and reward engine.

The reward layer is intentionally independent from environment execution.

Canonical model:

    RewardComponent
    RewardBreakdown
    RewardEngine

The engine validates reward values and component definitions before
producing a breakdown. Long-horizon tasks may use multiple components
per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping


_FLOAT_TOLERANCE = 1e-9


def _require_finite(value: float, field_name: str) -> float:
    """Require a finite numeric value."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite.")

    return value


@dataclass(frozen=True)
class RewardComponent:
    """One named reward contribution."""

    name: str
    value: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("component name must be a string.")

        if not self.name.strip():
            raise ValueError("component name must not be empty.")

        object.__setattr__(
            self,
            "value",
            _require_finite(self.value, "component value"),
        )

        object.__setattr__(
            self,
            "weight",
            _require_finite(self.weight, "component weight"),
        )

        if self.weight < 0:
            raise ValueError("component weight must be non-negative.")


@dataclass(frozen=True)
class RewardBreakdown:
    """Validated collection of reward components and their total."""

    components: tuple[RewardComponent, ...]
    total: float

    def __post_init__(self) -> None:
        names = [component.name for component in self.components]

        if len(names) != len(set(names)):
            raise ValueError("component names must be unique.")

        calculated_total = math.fsum(
            component.value * component.weight
            for component in self.components
        )

        validated_total = _require_finite(
            self.total,
            "reward total",
        )

        if not math.isclose(
            calculated_total,
            validated_total,
            rel_tol=_FLOAT_TOLERANCE,
            abs_tol=_FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "reward total does not equal the sum of weighted components."
            )

    @classmethod
    def from_components(
        cls,
        components: Iterable[RewardComponent],
    ) -> "RewardBreakdown":
        normalized = tuple(components)

        total = math.fsum(
            component.value * component.weight
            for component in normalized
        )

        return cls(
            components=normalized,
            total=total,
        )

    @property
    def weighted_components(self) -> Mapping[str, float]:
        """Return weighted contribution by component name."""
        return {
            component.name: component.value * component.weight
            for component in self.components
        }


class RewardEngine:
    """Canonical reward calculator for single-step or long-horizon rewards."""

    def __init__(
        self,
        *,
        tolerance: float = _FLOAT_TOLERANCE,
    ) -> None:
        tolerance = _require_finite(
            tolerance,
            "tolerance",
        )

        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")

        self._tolerance = tolerance

    @property
    def tolerance(self) -> float:
        return self._tolerance

    def calculate(
        self,
        components: Iterable[RewardComponent],
    ) -> RewardBreakdown:
        """Validate components and calculate their canonical breakdown."""
        normalized = tuple(components)

        names: set[str] = set()

        for component in normalized:
            if not isinstance(component, RewardComponent):
                raise TypeError(
                    "all reward components must be RewardComponent instances."
                )

            if component.name in names:
                raise ValueError(
                    f"duplicate reward component name: {component.name!r}"
                )

            names.add(component.name)

        total = math.fsum(
            component.value * component.weight
            for component in normalized
        )

        breakdown = RewardBreakdown(
            components=normalized,
            total=total,
        )

        recalculated = math.fsum(
            value
            for value in breakdown.weighted_components.values()
        )

        if not math.isclose(
            recalculated,
            breakdown.total,
            rel_tol=self._tolerance,
            abs_tol=self._tolerance,
        ):
            raise ValueError(
                "reward invariant violated: weighted components "
                "do not equal total."
            )

        return breakdown

    def compute(
        self,
        components: Iterable[RewardComponent],
    ) -> RewardBreakdown:
        """Compatibility alias for calculate()."""
        return self.calculate(components)

    def breakdown(
        self,
        components: Iterable[RewardComponent],
    ) -> RewardBreakdown:
        """Compatibility alias for calculate()."""
        return self.calculate(components)
