

class AntiCheatContract:
    """Contract-level anti-cheat boundary definitions."""

    AGENT_VISIBLE=frozenset({
        "observation","action","reward","terminated","truncated","info",
    })

    EVALUATOR_ONLY=frozenset({
        "hidden_scenario","oracle_state","reference_solution",
        "verifier_state","ground_truth","private_reward_components",
    })

    PROTECTED_PATHS=frozenset({
        "hidden_scenarios","oracle","verifier","ground_truth",
        "evaluator","private","reference_solution",
    })

    REWARD_HACKING_SIGNALS=frozenset({
        "reward_injection","reward_override","reward_bypass",
        "hidden_state_access","oracle_access","verifier_access",
        "ground_truth_access","protected_file_access",
    })

    @classmethod
    def is_agent_visible(cls,name):
        return name in cls.AGENT_VISIBLE

    @classmethod
    def is_evaluator_only(cls,name):
        return name in cls.EVALUATOR_ONLY

    @classmethod
    def is_protected_path(cls,name):
        return name in cls.PROTECTED_PATHS

    @classmethod
    def is_reward_hacking_signal(cls,name):
        return name in cls.REWARD_HACKING_SIGNALS


class AntiCheatEnforcer:
    "Runtime boundary checks for agent/evaluator separation."

    @classmethod
    def validate_agent_visible(cls, name):
        if AntiCheatContract.is_evaluator_only(name):
            raise PermissionError(
                f"Evaluator-only data cannot cross the agent boundary: {name}"
            )
        return AntiCheatContract.is_agent_visible(name)

    @classmethod
    def validate_evaluator_only(cls, name):
        if AntiCheatContract.is_agent_visible(name):
            raise PermissionError(
                f"Agent-visible data cannot be classified as evaluator-only: {name}"
            )
        return AntiCheatContract.is_evaluator_only(name)

    @classmethod
    def validate_path(cls, path):
        value=str(path).replace(chr(92), "/").lower()
        for protected in AntiCheatContract.PROTECTED_PATHS:
            if protected.lower() in value:
                raise PermissionError(
                    f"Protected evaluator path access denied: {path}"
                )
        return True

    @classmethod
    def validate_signal(cls, signal):
        if AntiCheatContract.is_reward_hacking_signal(signal):
            raise PermissionError(
                f"Reward-hacking signal detected: {signal}"
            )
        return True
