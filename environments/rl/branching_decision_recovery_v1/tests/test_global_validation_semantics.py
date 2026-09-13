from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv


def test_first_healthy_validation_is_local():
    env = BranchingDecisionRecoveryEnv()
    env.reset(seed=3001)

    _, reward, terminal, _ = env.step("validate")

    assert reward == 0.75
    assert terminal is False
    assert env.state is not None
    assert env.state.validation_status == "local_validated"


def test_second_healthy_validation_promotes_to_global():
    env = BranchingDecisionRecoveryEnv()
    env.reset(seed=3001)

    env.step("validate")
    _, reward, terminal, _ = env.step("validate")

    assert reward == 1.0
    assert terminal is False
    assert env.state is not None
    assert env.state.validation_status == "globally_validated"


def test_failure_cannot_be_globally_validated():
    env = BranchingDecisionRecoveryEnv()
    env.reset(seed=3001)

    assert env.state is not None
    env.state.failure_detected = True

    env.step("validate")
    env.step("validate")

    assert env.state.validation_status == "local_validated"
    assert env.state.failure_detected is True
