from agentforge.security.anti_cheat import AntiCheatContract, AntiCheatEnforcer
import pytest


def test_agent_visible_allowed():
    assert AntiCheatEnforcer.validate_agent_visible("observation")
    assert AntiCheatEnforcer.validate_agent_visible("action")


def test_evaluator_only_rejected_at_agent_boundary():
    with pytest.raises(PermissionError):
        AntiCheatEnforcer.validate_agent_visible("hidden_scenario")


def test_agent_visible_rejected_as_evaluator_only():
    with pytest.raises(PermissionError):
        AntiCheatEnforcer.validate_evaluator_only("observation")


def test_evaluator_only_allowed():
    assert AntiCheatEnforcer.validate_evaluator_only("oracle_state")


def test_protected_path_rejected():
    with pytest.raises(PermissionError):
        AntiCheatEnforcer.validate_path("/app/hidden_scenarios/case.json")


def test_normal_path_allowed():
    assert AntiCheatEnforcer.validate_path("/app/public/scenarios/case.json")


def test_reward_hacking_signal_rejected():
    with pytest.raises(PermissionError):
        AntiCheatEnforcer.validate_signal("reward_override")


def test_normal_signal_allowed():
    assert AntiCheatEnforcer.validate_signal("normal_transition")
