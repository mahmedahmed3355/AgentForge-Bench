from agentforge import LongHorizonTaskRequirements


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
