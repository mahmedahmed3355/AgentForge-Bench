from data_pipeline_recovery_v1.decision_graph import IncidentKind
from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv
from data_pipeline_recovery_v1.report import build_agent_report


def test_report_captures_initial_failure_state():
    env = DataPipelineRecoveryEnv(incident=IncidentKind.SCHEMA)

    report = build_agent_report(
        task_id="pipeline-v1-test-001",
        state=env.state,
        total_reward=0.0,
        last_action="inspect_pipeline",
    )

    assert report.status == "FAILED"
    assert report.actions == 0
    assert report.logical_stages == 0
    assert report.incident == "schema"
    assert report.terminal is False
    assert report.success is False
    assert report.failure_reason == "incomplete"


def test_report_captures_success_state():
    env = DataPipelineRecoveryEnv(incident=IncidentKind.SCHEMA)

    env.state.incident_observed = True
    env.state.diagnosis_started = True
    env.state.root_cause_identified = True
    env.state.selected_branch = "schema_recovery"
    env.state.selected_strategy = "repair_upstream"
    env.state.pipeline_run_completed = True
    env.state.recovery_completed = True
    env.state.global_validation_completed = True
    env.state.terminal = True
    env.state.success = True
    env.state.action_count = 14
    env.state.logical_stage = 14

    report = build_agent_report(
        task_id="pipeline-v1-test-success",
        state=env.state,
        total_reward=25.5,
        last_action="final_verify",
    )

    assert report.status == "PASSED"
    assert report.actions == 14
    assert report.logical_stages == 14
    assert report.total_reward == 25.5
    assert report.terminal is True
    assert report.success is True
    assert "Final verification" in report.solved
    assert report.failure_reason == ""

    rendered = report.render()

    assert "Status: PASSED" in rendered
    assert "Actions: 14" in rendered
    assert "Logical stages: 14" in rendered
    assert "Total reward: 25.50" in rendered


def test_report_detects_downstream_failure():
    env = DataPipelineRecoveryEnv(
        incident=IncidentKind.TRANSFORMATION
    )

    env.state.action_count = 11
    env.state.logical_stage = 13
    env.state.downstream_inconsistency = True
    env.state.recovery_required = True

    report = build_agent_report(
        task_id="pipeline-v1-test-failure",
        state=env.state,
        total_reward=-2.0,
        last_action="run_pipeline",
    )

    assert report.status == "FAILED"
    assert report.actions == 11
    assert report.logical_stages == 13
    assert report.failure_reason == "downstream_inconsistency"
    assert report.stopping_point == "run_pipeline"
    assert report.downstream_inconsistency is True
