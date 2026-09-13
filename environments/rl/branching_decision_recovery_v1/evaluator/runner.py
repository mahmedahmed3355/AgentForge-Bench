from __future__ import annotations

import json

from .harness import run_hidden_evaluation


def main() -> int:
    report = run_hidden_evaluation()

    print(json.dumps(report, indent=2, sort_keys=True))

    return 0 if report["hidden_evaluation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
