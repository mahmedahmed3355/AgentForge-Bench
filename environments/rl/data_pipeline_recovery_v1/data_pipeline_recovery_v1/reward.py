from __future__ import annotations


class PipelineRecoveryReward:
    """
    Reward policy for long-horizon recovery.

    Final global verification dominates local milestone rewards.
    """

    FINAL_SUCCESS = 10.0
    FINAL_FAILURE = -5.0

    MILESTONE_INSPECTION = 1.0
    MILESTONE_DIAGNOSIS = 1.5
    MILESTONE_REPAIR = 1.25
    MILESTONE_RECOVERY = 2.0
    MILESTONE_REPLAN = 1.5

    INVALID_ACTION = -1.0
    DOWNSTREAM_FAILURE = -2.0
