from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FailureDiagnostics:
    failed: bool
    failures: tuple[str, ...]


def diagnose_failures(
    checks: Iterable[tuple[str, bool]],
) -> FailureDiagnostics:
    failures = tuple(
        name
        for name, passed in checks
        if not passed
    )

    return FailureDiagnostics(
        failed=bool(failures),
        failures=failures,
    )
