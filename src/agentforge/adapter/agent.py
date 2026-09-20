from __future__ import annotations

from typing import Any, Protocol


class AgentProtocol(Protocol):
    def act(self, observation: Any) -> Any:
        ...


class AgentAdapter:
    """Minimal adapter around an AgentForge-compatible agent."""

    def __init__(self, agent: AgentProtocol) -> None:
        self.agent = agent

    def act(self, observation: Any) -> Any:
        return self.agent.act(observation)
