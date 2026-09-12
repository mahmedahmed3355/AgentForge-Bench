# Task 002 — Reward Model

The reward model is intended to encourage useful progress while penalizing
invalid transitions and failed execution.

Current public reward categories include:

- inspection progress
- diagnosis
- branch selection
- strategy selection
- local repair
- configuration changes
- pipeline execution
- validation
- recovery
- replanning
- final verification

Negative rewards are used for invalid prerequisites and failed execution.

## Design Requirement

Reward must not become a substitute for correctness.

An agent should not be able to obtain a successful terminal state by repeatedly
performing rewarded local actions while leaving the pipeline globally invalid.

The final success predicate therefore depends on actual state:

- diagnosis completed
- valid branch selected
- valid strategy selected
- execution completed
- all required components valid
- output validation completed
- no unresolved downstream inconsistency
- no unresolved recovery requirement
- no unresolved replanning requirement

Future versions should add stronger reward-hacking tests before freezing.
