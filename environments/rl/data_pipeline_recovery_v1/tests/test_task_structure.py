from pathlib import Path

from data_pipeline_recovery_v1.actions import ACTION_NAMES
from data_pipeline_recovery_v1.decision_graph import BRANCHES, IncidentKind
from data_pipeline_recovery_v1.state import PIPELINE_COMPONENTS, PipelineState


ROOT = Path(__file__).resolve().parents[1]


def test_all_required_task_directories_exist():
    assert (ROOT / "data").is_dir()
    assert (ROOT / "docs").is_dir()
    assert (ROOT / "evaluator").is_dir()
    assert (ROOT / "data_pipeline_recovery_v1").is_dir()


def test_branching_has_multiple_long_horizon_paths():
    assert len(BRANCHES) >= 4

    for branch in BRANCHES:
        assert branch.minimum_logical_stages >= 14
        assert len(branch.strategies) >= 2


def test_incident_families_are_distinct():
    incidents = {branch.incident for branch in BRANCHES}

    assert incidents == {
        IncidentKind.SCHEMA,
        IncidentKind.TRANSFORMATION,
        IncidentKind.DATA_QUALITY,
        IncidentKind.RESOURCE,
    }


def test_actions_include_diagnosis_recovery_and_replanning():
    required = {
        "inspect_pipeline",
        "inspect_component",
        "analyze_failure",
        "select_branch",
        "select_strategy",
        "modify_component",
        "run_pipeline",
        "recover",
        "replan",
        "final_verify",
    }

    assert required.issubset(set(ACTION_NAMES))


def test_pipeline_state_is_stateful():
    state = PipelineState(episode_id="test-001")

    assert state.logical_stage == 0
    assert state.action_count == 0
    assert not state.terminal
    assert not state.success
    assert not state.all_components_valid

    state.schema_valid = True
    state.cleaning_valid = True
    state.transformation_valid = True
    state.aggregation_valid = True
    state.output_valid = True

    assert state.all_components_valid
