from branching_decision_recovery_v1.actions import ActionKind
from branching_decision_recovery_v1.decision_graph import DecisionBranch
from branching_decision_recovery_v1.inspect_adapter import InspectAdapter
from branching_decision_recovery_v1.state import DecisionState


def test_core_imports():
    assert ActionKind.INSPECT_SYSTEM.value == "inspect_system"
    assert DecisionBranch.BRANCH_A.value == "branch_a"


def test_state():
    state = DecisionState()
    assert state.logical_stage == 0
    assert state.terminal is False


def test_inspect_adapter():
    state = DecisionState()
    state.system_status = "unstable"
    result = InspectAdapter().inspect_system(state)
    assert result.information_key == "system_status"
    assert result.information_value == "unstable"
