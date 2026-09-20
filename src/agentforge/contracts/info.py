"""Agent-visible info contract."""

from dataclasses import dataclass
from typing import Any


FORBIDDEN_KEYS = frozenset(
    {
        "oracle_action",
        "oracle_trajectory",
        "hidden_solution",
        "ground_truth",
        "reference_trajectory",
        "verifier_answer",
        "correct_action",
    }
)


@dataclass(frozen=True)
class InfoContract:
    """Validates evaluator-safe info dictionaries."""

    forbidden_keys: frozenset[str] = FORBIDDEN_KEYS

    def validate(self, info: dict[str, Any]) -> None:
        if not isinstance(info, dict):
            raise TypeError("info must be a dictionary.")

        leaked = set(info).intersection(self.forbidden_keys)

        if leaked:
            names = ", ".join(sorted(leaked))
            raise ValueError(
                f"Agent-visible info contains forbidden evaluator data: {names}"
            )
