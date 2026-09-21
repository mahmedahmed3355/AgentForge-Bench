from importlib import import_module

CANONICAL_APIS = {
    "TaskSpec": "agentforge.contracts.task",
    "AgentForgeEnv": "agentforge.environment.base",
    "GymnasiumAdapter": "agentforge.environment.gym_adapter",
    "EpisodeRunner": "agentforge.runtime.episode_runner",
    "EvaluationRunner": "agentforge.runtime.evaluation_runner",
    "Trajectory": "agentforge.runtime.trajectory",
    "TrajectoryStep": "agentforge.runtime.trajectory",
    "TrajectoryRecorder": "agentforge.runtime.trajectory_recorder",
    "RewardEngine": "agentforge.runtime.contracts.rewards",
    "BaseVerifier": "agentforge.verifier.contracts",
    "VerifierExecutor": "agentforge.verifier.executable",
    "RLAgent": "agentforge.agents.rl.base",
    "Policy": "agentforge.agents.rl.policy",
}

def test_canonical_api_ownership():
    for symbol, module_name in CANONICAL_APIS.items():
        module = import_module(module_name)
        assert hasattr(module, symbol), f"{symbol} missing from {module_name}"
        assert getattr(module, symbol).__module__ == module_name

def test_legacy_gymnasium_adapter_reexports_canonical():
    canonical = import_module("agentforge.environment.gym_adapter").GymnasiumAdapter
    legacy = import_module("agentforge.adapter.gymnasium").GymnasiumAdapter
    assert legacy is canonical
