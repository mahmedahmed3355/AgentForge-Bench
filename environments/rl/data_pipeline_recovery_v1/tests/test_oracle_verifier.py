from data_pipeline_recovery_v1.decision_graph import IncidentKind, RecoveryStrategy
from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv
from data_pipeline_recovery_v1.oracle import PipelineRecoveryOracle
from data_pipeline_recovery_v1.verifier import PipelineRecoveryVerifier


def test_oracle_has_reference_policy_for_all_incidents():
    oracle = PipelineRecoveryOracle()

    for incident in IncidentKind:
        result = oracle.expected(incident)
        assert result.success
        assert result.expected_branch
        assert result.expected_strategy
        assert result.minimum_logical_stage >= 14


def test_verifier_rejects_non_terminal_state():
    env = DataPipelineRecoveryEnv(
        episode_id="verifier-negative",
        incident=IncidentKind.SCHEMA,
    )
    env.reset()

    result = PipelineRecoveryVerifier().verify(env.state)

    assert not result.success
    assert not result.terminal


def test_oracle_rejects_wrong_strategy():
    oracle = PipelineRecoveryOracle()

    result = oracle.evaluate(
        IncidentKind.SCHEMA,
        branch="schema_recovery",
        strategy=RecoveryStrategy.REPAIR_CONSUMER,
        logical_stage=20,
        all_components_valid=True,
        output_valid=True,
        recovery_required=False,
        replanning_required=False,
        downstream_inconsistency=False,
        pipeline_run_completed=True,
    )

    assert not result.success
