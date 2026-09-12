# CUDA Kernel Optimization V1 Task Contract

This task is a long-horizon reinforcement-learning environment.

## Mandatory Properties

The task must provide:

- 20 logical stages
- Persistent state
- Branching decisions
- Decision-dependent consequences
- Delayed consequences
- Reasoning-dependent action selection
- Failure detection
- Recovery
- Replanning
- Hidden evaluation
- Unseen scenarios
- Oracle validation
- Independent verification
- Reward-hacking resistance
- Prime Verifiers V1 compatibility
- RL training readiness

## Logical Horizon

Twenty logical stages do not imply twenty agent actions.

The agent may require more than twenty actions because:

- an action can fail
- a branch can require additional inspection
- compilation can fail
- correctness can fail
- a performance regression can require recovery
- the agent may need to re-plan

The environment must therefore model logical progress separately from raw action count.

## Branching Requirement

At least one meaningful decision must expose multiple viable or plausible paths.

Different choices must be capable of producing different downstream states or consequences.

A task must not be a fixed action script disguised as a branching environment.

## Recovery Requirement

A wrong or suboptimal decision must be detectable through environment feedback.

The agent must have an opportunity to:

1. Observe the consequence.
2. Diagnose the failure.
3. Select an alternative.
4. Recover.
5. Continue toward the terminal objective.

## Reasoning Requirement

The environment should require decisions based on accumulated observations and consequences.

The benchmark does not require exposing or collecting private chain-of-thought.

Evaluation is based on observable behavior and final state.

## Evaluation Isolation

The following must remain outside agent-visible observations:

- hidden scenarios
- hidden tests
- reference solutions
- verifier internals
- hidden performance thresholds
- evaluation-only metadata

## Production Gate

The current implementation is a structural baseline.

Production readiness additionally requires:

- real CUDA execution
- isolated workspace
- real compilation
- randomized correctness validation
- independent benchmarking
- timeout and resource controls
- hidden runtime evaluation
- trace validation
- RL rollout smoke testing
