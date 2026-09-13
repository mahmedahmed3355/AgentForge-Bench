from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv
from branching_decision_recovery_v1.gymnasium_env import (
    BranchingDecisionRecoveryGymEnv,
)
from branching_decision_recovery_v1.observations import (
    Observation,
    inspect_system,
    validate_observation_contract,
)


def test_observation_does_not_expose_hidden_truth():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    result = inspect_system(env.state)

    assert result.value
    assert "hidden_information" not in result.details
    assert "ground_truth" not in result.details
    assert "oracle_answer" not in result.details


def test_inspection_result_has_information_value_alias():
    env = BranchingDecisionRecoveryEnv()
    env.reset()

    result = inspect_system(env.state)

    assert result.information_value == result.value


def test_observation_contract_accepts_public_state():
    observation = Observation(
        step=0,
        logical_stage=0,
        system_status="degraded",
        current_component=None,
        visible_components=("source", "processing", "output"),
        available_actions=("inspect_system", "select_branch"),
        observed_information=(),
        warnings=("additional_information_required",),
        local_metrics={},
        hypothesis=None,
        hypothesis_confidence=0.0,
        selected_branch=None,
        selected_strategy=None,
        failure_detected=False,
        recovery_required=False,
        replanning_required=False,
        terminal=False,
        success=False,
    )

    validate_observation_contract(observation)


def test_gymnasium_reset_returns_valid_observation():
    env = BranchingDecisionRecoveryGymEnv()

    observation, info = env.reset(seed=3001)

    assert isinstance(observation, dict)
    assert info["seed"] == 3001
    assert observation["terminal"] == 0
    assert observation["success"] == 0


def test_gymnasium_step_returns_standard_tuple():
    env = BranchingDecisionRecoveryGymEnv()
    env.reset(seed=3001)

    observation, reward, terminated, truncated, info = env.step(0)

    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_gymnasium_reset_is_deterministic_for_seed():
    env1 = BranchingDecisionRecoveryGymEnv()
    env2 = BranchingDecisionRecoveryGymEnv()

    obs1, info1 = env1.reset(seed=3011)
    obs2, info2 = env2.reset(seed=3011)

    assert info1 == info2
    assert obs1 == obs2
