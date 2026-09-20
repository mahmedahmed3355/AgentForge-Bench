"""Environment lifecycle contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LifecycleContract:
    """Canonical lifecycle invariants."""

    requires_reset_before_step: bool = True
    requires_reset_after_terminal: bool = True
    allows_implicit_reset: bool = False
    episode_id_required: bool = True

    def validate_flags(self, terminated: bool, truncated: bool) -> None:
        if terminated and truncated:
            raise ValueError(
                "terminated and truncated cannot both be True"
            )
