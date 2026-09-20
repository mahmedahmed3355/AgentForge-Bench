from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleComparison:
    oracle_reward: float
    agent_reward: float

    @property
    def reward_gap(self) -> float:
        return self.oracle_reward - self.agent_reward

    @property
    def exact_match(self) -> bool:
        return self.oracle_reward == self.agent_reward
