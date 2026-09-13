from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


__all__ = [
    "PublicScenario",
    "load_public_scenarios",
]


@dataclass(frozen=True)
class PublicScenario:
    scenario_id: str
    seed: int
    scenario_family: str


def _scenario_data_path() -> Path:
    package_data = (
        Path(__file__).resolve().parent
        / "data"
        / "scenarios.json"
    )

    if package_data.is_file():
        return package_data

    source_data = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "scenarios.json"
    )

    if source_data.is_file():
        return source_data

    raise FileNotFoundError(
        "Public scenario data not found in package or source layout."
    )


def load_public_scenarios() -> tuple[PublicScenario, ...]:
    data_path = _scenario_data_path()

    with data_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    scenarios = payload.get("scenarios", [])

    return tuple(
        PublicScenario(
            scenario_id=str(item["scenario_id"]),
            seed=int(item["seed"]),
            scenario_family=str(item["scenario_family"]),
        )
        for item in scenarios
    )
