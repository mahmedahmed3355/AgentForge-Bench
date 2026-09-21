import importlib

CANONICAL = {
    "TaskSpec": ("agentforge.contracts.task", "TaskSpec"),
    "AgentForgeEnv": ("agentforge.environment.base", "AgentForgeEnv"),
    "GymnasiumAdapter": ("agentforge.environment.gym_adapter", "GymnasiumAdapter"),
    "EpisodeRunner": ("agentforge.runtime.episode_runner", "EpisodeRunner"),
    "EvaluationRunner": ("agentforge.runtime.evaluation_runner", "EvaluationRunner"),
    "Trajectory": ("agentforge.runtime.trajectory", "Trajectory"),
    "TrajectoryStep": ("agentforge.runtime.trajectory", "TrajectoryStep"),
    "TrajectoryRecorder": ("agentforge.runtime.trajectory_recorder", "TrajectoryRecorder"),
    "RewardEngine": ("agentforge.runtime.contracts.rewards", "RewardEngine"),
    "BaseVerifier": ("agentforge.verifier.contracts", "BaseVerifier"),
    "VerifierExecutor": ("agentforge.verifier.executable", "VerifierExecutor"),
    "RLAgent": ("agentforge.agents.rl.base", "RLAgent"),
    "Policy": ("agentforge.agents.rl.policy", "Policy"),
}

def test_canonical_modules_import():
    for name, (module_name, attribute) in CANONICAL.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), name

def test_canonical_module_ownership():
    for name, (module_name, attribute) in CANONICAL.items():
        module = importlib.import_module(module_name)
        symbol = getattr(module, attribute)
        assert symbol.__module__ == module_name, (name, symbol.__module__, module_name)
