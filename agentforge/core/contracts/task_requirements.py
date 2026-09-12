from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongHorizonTaskRequirements:
    logical_stages: int
    stateful: bool
    branching: bool
    decision_making: bool
    delayed_consequences: bool
    recovery: bool
    reasoning_required: bool
    hidden_evaluation: bool
    unseen_scenarios: bool
    oracle_required: bool
    independent_verifier: bool
    reward_hacking_resistant: bool
    prime_v1_compatible: bool
    rl_training_ready: bool

    def validate(self) -> None:
        if self.logical_stages < 10:
            raise ValueError("Long-horizon tasks require at least 10 logical stages.")

        requirements = {
            "stateful": self.stateful,
            "branching": self.branching,
            "decision_making": self.decision_making,
            "delayed_consequences": self.delayed_consequences,
            "recovery": self.recovery,
            "reasoning_required": self.reasoning_required,
            "hidden_evaluation": self.hidden_evaluation,
            "unseen_scenarios": self.unseen_scenarios,
            "oracle_required": self.oracle_required,
            "independent_verifier": self.independent_verifier,
            "reward_hacking_resistant": self.reward_hacking_resistant,
            "prime_v1_compatible": self.prime_v1_compatible,
            "rl_training_ready": self.rl_training_ready,
        }

        missing = [
            name
            for name, enabled in requirements.items()
            if not enabled
        ]

        if missing:
            raise ValueError(
                "Task does not satisfy mandatory AgentForge RL requirements: "
                + ", ".join(missing)
            )
