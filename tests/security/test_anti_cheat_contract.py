from agentforge.security.anti_cheat import AntiCheatContract


def test_agent_visible_contract():
    assert "observation" in AntiCheatContract.AGENT_VISIBLE
    assert "action" in AntiCheatContract.AGENT_VISIBLE
    assert "reward" in AntiCheatContract.AGENT_VISIBLE


def test_evaluator_only_contract():
    assert "hidden_scenario" in AntiCheatContract.EVALUATOR_ONLY
    assert "oracle_state" in AntiCheatContract.EVALUATOR_ONLY
    assert "ground_truth" in AntiCheatContract.EVALUATOR_ONLY


def test_protected_paths_contract():
    assert "hidden_scenarios" in AntiCheatContract.PROTECTED_PATHS
    assert "oracle" in AntiCheatContract.PROTECTED_PATHS
    assert "verifier" in AntiCheatContract.PROTECTED_PATHS


def test_reward_hacking_signals_contract():
    assert "reward_injection" in AntiCheatContract.REWARD_HACKING_SIGNALS
    assert "reward_override" in AntiCheatContract.REWARD_HACKING_SIGNALS
    assert "oracle_access" in AntiCheatContract.REWARD_HACKING_SIGNALS


def test_boundary_helpers():
    assert AntiCheatContract.is_agent_visible("observation")
    assert AntiCheatContract.is_evaluator_only("hidden_scenario")
    assert AntiCheatContract.is_protected_path("oracle")
    assert AntiCheatContract.is_reward_hacking_signal("reward_override")


def test_boundary_separation():
    assert AntiCheatContract.AGENT_VISIBLE.isdisjoint(
        AntiCheatContract.EVALUATOR_ONLY
    )
