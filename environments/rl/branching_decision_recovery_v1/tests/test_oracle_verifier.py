from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv
from branching_decision_recovery_v1.oracle import BranchingDecisionOracle
from branching_decision_recovery_v1.verifier import BranchingDecisionVerifier


def test_oracle_and_verifier_accept_valid_terminal_state():
    env = BranchingDecisionRecoveryEnv()
    oracle = BranchingDecisionOracle()
    verifier = BranchingDecisionVerifier()

    env.reset(seed=3001)

    env.state.selected_branch = "branch_b"
    env.state.selected_strategy = "strategy_b"
    env.state.logical_stage = 12
    env.state.failure_detected = False
    env.state.recovery_required = False
    env.state.replanning_required = False
    env.state.validation_status = "globally_validated"
    env.state.terminal = True
    env.state.success = True

    oracle_result = oracle.evaluate(env.state)
    verifier_result = verifier.verify(env.state)

    assert oracle_result.solved is True
    assert verifier_result.verified is True


def test_verifier_rejects_active_failure():
    env = BranchingDecisionRecoveryEnv()
    verifier = BranchingDecisionVerifier()

    env.reset(seed=3001)

    env.state.selected_branch = "branch_b"
    env.state.selected_strategy = "strategy_b"
    env.state.logical_stage = 12
    env.state.failure_detected = True
    env.state.validation_status = "globally_validated"
    env.state.terminal = True
    env.state.success = True

    result = verifier.verify(env.state)

    assert result.verified is False
    assert "active failure" in result.reason


def test_verifier_rejects_local_validation_only():
    env = BranchingDecisionRecoveryEnv()
    verifier = BranchingDecisionVerifier()

    env.reset(seed=3001)

    env.state.selected_branch = "branch_b"
    env.state.selected_strategy = "strategy_b"
    env.state.logical_stage = 12
    env.state.failure_detected = False
    env.state.validation_status = "local_validated"
    env.state.terminal = True
    env.state.success = True

    result = verifier.verify(env.state)

    assert result.verified is False
    assert "global validation" in result.reason


def test_oracle_preserves_failure_history_as_nonterminal_evidence():
    env = BranchingDecisionRecoveryEnv()
    oracle = BranchingDecisionOracle()

    env.reset(seed=3001)

    env.state.selected_branch = "branch_b"
    env.state.selected_strategy = "strategy_b"
    env.state.logical_stage = 12
    env.state.failure_detected = False
    env.state.recovery_required = False
    env.state.replanning_required = False
    env.state.validation_status = "globally_validated"
    env.state.terminal = True
    env.state.success = True
    env.state.failure_history.append(
        {
            "source": "downstream_dependency",
            "effect": "historical delayed failure",
        }
    )

    result = oracle.evaluate(env.state)

    assert result.solved is True
    assert len(env.state.failure_history) == 1
