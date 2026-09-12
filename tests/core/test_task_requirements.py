import pytest

from agentforge import LongHorizonTaskRequirements


def make_valid_requirements():
    return LongHorizonTaskRequirements(
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


def test_valid_requirements():
    make_valid_requirements().validate()


def test_short_horizon_is_rejected():
    requirements = make_valid_requirements()

    invalid = LongHorizonTaskRequirements(
        logical_stages=5,
        stateful=requirements.stateful,
        branching=requirements.branching,
        decision_making=requirements.decision_making,
        delayed_consequences=requirements.delayed_consequences,
        recovery=requirements.recovery,
        reasoning_required=requirements.reasoning_required,
        hidden_evaluation=requirements.hidden_evaluation,
        unseen_scenarios=requirements.unseen_scenarios,
        oracle_required=requirements.oracle_required,
        independent_verifier=requirements.independent_verifier,
        reward_hacking_resistant=requirements.reward_hacking_resistant,
        prime_v1_compatible=requirements.prime_v1_compatible,
        rl_training_ready=requirements.rl_training_ready,
    )

    with pytest.raises(ValueError):
        invalid.validate()


def test_missing_branching_is_rejected():
    requirements = make_valid_requirements()

    invalid = LongHorizonTaskRequirements(
        logical_stages=requirements.logical_stages,
        stateful=True,
        branching=False,
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

    with pytest.raises(ValueError):
        invalid.validate()
