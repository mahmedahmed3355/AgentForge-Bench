"""Action contract."""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ActionContract:
    """Defines validation and serialization rules for agent actions."""

    validator: Callable[[Any], bool]
    serializer: Callable[[Any], Any]

    def validate(self, action: Any) -> None:
        if not self.validator(action):
            raise ValueError("Action violates action contract.")

    def serialize(self, action: Any) -> Any:
        self.validate(action)
        return self.serializer(action)
