from __future__ import annotations

from enum import StrEnum


class KernelAction(StrEnum):
    INSPECT = "inspect"
    MODIFY = "modify"
    COMPILE = "compile"
    CHECK_CORRECTNESS = "check_correctness"
    BENCHMARK = "benchmark"
    ANALYZE = "analyze"
    RESET_CANDIDATE = "reset_candidate"
    FINAL_VERIFY = "final_verify"


VALID_ACTIONS = tuple(action.value for action in KernelAction)
