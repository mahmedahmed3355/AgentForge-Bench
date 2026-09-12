# AgentForge-Bench Core Contracts

This document defines the framework-level contracts shared by AgentForge-Bench environments.

## TaskData

TaskData is the immutable description of a benchmark task.

It contains:

- task index
- task name
- description
- prompt
- metadata

TaskData must not contain hidden evaluation truth.

## Task

Task owns the executable task behavior.

A Task is responsible for:

- scenario initialization
- environment state
- observations
- action execution
- termination
- truncation
- task-local behavior

## Taskset

Taskset is the collection boundary for benchmark tasks.

It loads TaskData and provides the task population used by training or evaluation.

## State

Environment state represents the true operational state of the task.

Private evaluation truth must remain isolated from agent-visible observations.

## Action

Action represents an agent operation and its parameters.

Actions must have explicit semantics and deterministic validation.

## Observation

Observation is the information exposed to the agent.

Observations must not leak:

- oracle solutions
- hidden test data
- hidden scenario parameters
- verifier internals
- evaluation truth

## Reward

Reward is the learning signal.

Reward may contain:

- progress
- milestone
- recovery
- correctness
- performance
- efficiency
- final success

Reward design must resist reward hacking.

## Verifier

The verifier independently determines whether the task outcome satisfies the required contract.

Agent self-reported success is never sufficient.

## Oracle

The oracle establishes reference behavior and validates that the task is solvable.

Oracle validation is a prerequisite for difficulty calibration.

## Scenario

Scenario defines an episode instance and its reproducible seed.

Supported splits include:

- train
- eval
- hidden

## Trace

Trace records the interaction trajectory.

A trace should preserve enough information for:

- debugging
- evaluation
- analysis
- RL training
- reproducibility

## Prime V1 Boundary

These framework contracts are AgentForge abstractions.

Prime V1 integration is implemented as an adapter/environment layer rather than leaking Prime-specific implementation details throughout the core framework.
