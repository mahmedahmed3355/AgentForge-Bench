"""Reproducibility and seed contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedContract:
    """Defines canonical environment seeding semantics."""

    deterministic: bool = True
    seed_required_on_reset_only: bool = True

    def validate(self, seed: int | None) -> None:
        if seed is not None and not isinstance(seed, int):
            raise TypeError("seed must be an int or None.")
