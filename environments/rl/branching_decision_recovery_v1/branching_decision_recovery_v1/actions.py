from __future__ import annotations

from enum import Enum


class ActionKind(str, Enum):
    INSPECT_SYSTEM = "inspect_system"
    INSPECT_COMPONENT = "inspect_component"
    INSPECT_DEPENDENCY = "inspect_dependency"
    INSPECT_HISTORY = "inspect_history"
    PROBE_STATE = "probe_state"
    QUERY_VALIDATION = "query_validation"

    FORM_HYPOTHESIS = "form_hypothesis"

    SELECT_BRANCH = "select_branch"
    SELECT_STRATEGY = "select_strategy"

    APPLY_ACTION = "apply_action"
    MODIFY_CONFIGURATION = "modify_configuration"

    RUN = "run"
    VALIDATE = "validate"

    RECOVER = "recover"
    REPLAN = "replan"

    FINAL_VERIFY = "final_verify"
