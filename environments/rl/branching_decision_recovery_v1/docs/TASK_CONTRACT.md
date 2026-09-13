# Task 003 Contract

## Required Capability

The environment must test decision-making under uncertainty rather than simple execution.

## Observation Contract

Observations must expose enough information for rational interaction while withholding hidden evaluation truth.

The observation must not directly expose:

- correct branch
- optimal strategy
- hidden root cause
- hidden expected trajectory
- oracle answer
- reference solution

## Action Contract

Every action must have:

- explicit semantics
- preconditions
- state transition behavior
- observable result
- reward semantics where applicable

Invalid actions must not become a reward-farming mechanism.

## Branch Contract

Each scenario must contain multiple plausible choices.

At least one branch should require later validation to distinguish it from alternatives.

## Delayed Consequence Contract

At least one meaningful scenario must contain a causal relationship where an earlier decision causes a later observable consequence.

## Recovery Contract

Recovery must preserve relevant historical state.

Recovery must not be equivalent to environment reset.

## Replanning Contract

At least one path must require the agent to revise a previous plan after new information or failure.

## Terminal Contract

Success requires global correctness and final verification.

Local validation alone must never imply terminal success.

## Oracle Contract

The Oracle represents reference-valid behavior and terminal requirements.

It must support correct, incorrect, partial, and recovery trajectories.

## Verifier Contract

The Verifier independently determines whether the terminal state satisfies the task contract.

The verifier must not depend on hidden agent reasoning.

## Evaluation Contract

The benchmark must support:

- public scenarios
- hidden scenarios
- unseen scenario variation
- deterministic reset where required
- reproducibility
- anti-cheat validation
- reward-hacking audits
- clean-room execution
- agent calibration
