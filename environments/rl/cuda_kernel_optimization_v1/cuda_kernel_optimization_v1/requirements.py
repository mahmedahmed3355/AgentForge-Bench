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
            raise ValueError("logical_stages must be >= 10")

        required_flags = {
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
            name for name, enabled in required_flags.items()
            if not enabled
        ]

        if missing:
            raise ValueError(
                "Long-horizon requirements not satisfied: "
                + ", ".join(missing)
            )



CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS = LongHorizonTaskRequirements(
    logical_stages=20,
    stateful=True,
    branching=True,
    decision_making=True,
    delayed_consequences=True,
    recovery=True,
    reasoning_required=True,
    hidden_evaluation=True,
    unseen_scenarios=True,
    oracle_required=True,
    independent_verifier=True,
    reward_hacking_resistant=True,
    prime_v1_compatible=True,
    rl_training_ready=True,
)


def validate_requirements() -> None:
    CUDA_KERNEL_OPTIMIZATION_REQUIREMENTS.validate()
