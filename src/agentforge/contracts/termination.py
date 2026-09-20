"""Termination and truncation contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminationContract:
    """Separates task termination from external truncation."""

    def validate(self, terminated: bool, truncated: bool) -> None:
        if not isinstance(terminated, bool):
            raise TypeError("terminated must be bool.")

        if not isinstance(truncated, bool):
            raise TypeError("truncated must be bool.")

        if terminated and truncated:
            raise ValueError(
                "terminated and truncated cannot both be True."
            )
