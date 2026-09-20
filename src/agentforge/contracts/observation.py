"""Observation contract."""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ObservationContract:
    """Defines validation and serialization rules for observations."""

    validator: Callable[[Any], bool]
    serializer: Callable[[Any], Any]

    def validate(self, observation: Any) -> None:
        if not self.validator(observation):
            raise ValueError("Observation violates observation contract.")

    def serialize(self, observation: Any) -> Any:
        self.validate(observation)
        return self.serializer(observation)
