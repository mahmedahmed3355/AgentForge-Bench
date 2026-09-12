from __future__ import annotations

from enum import Enum


class ActionKind(str, Enum):
    INSPECT_PIPELINE = "inspect_pipeline"
    INSPECT_COMPONENT = "inspect_component"
    ANALYZE_FAILURE = "analyze_failure"

    SELECT_BRANCH = "select_branch"
    SELECT_STRATEGY = "select_strategy"

    MODIFY_COMPONENT = "modify_component"
    MODIFY_CONFIG = "modify_config"

    RUN_PIPELINE = "run_pipeline"
    CHECK_SCHEMA = "check_schema"
    CHECK_OUTPUT = "check_output"

    RECOVER = "recover"
    REPLAN = "replan"

    FINAL_VERIFY = "final_verify"


ACTION_NAMES = tuple(action.value for action in ActionKind)
