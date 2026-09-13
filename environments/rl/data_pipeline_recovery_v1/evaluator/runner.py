from __future__ import annotations

from .harness import run_hidden_evaluation


def main() -> int:
    results = run_hidden_evaluation()

    for result in results:
        print(
            f"{result.scenario_id}: "
            f"oracle={result.oracle_success} "
            f"verifier={result.verifier_success} "
            f"terminal={result.terminal} "
            f"stage={result.logical_stage}"
        )

    passed = all(
        result.oracle_success
        and result.verifier_success
        and result.terminal
        for result in results
    )

    print(f"hidden_evaluation_pass={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
