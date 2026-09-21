# AgentForge-RL-Bench — Task Authoring Contract

## Purpose

Task authoring defines the canonical declarative specification used to describe an RL benchmark task.

The canonical task model is `TaskSpec`.

## TaskSpec

A `TaskSpec` contains:

- `identity`
- `environment`
- `scenario`
- `actions`
- `observations`
- `reward`
- `termination`
- `success`
- `oracle`
- `verifier`
- `compatibility`
- `objective`
- `difficulty`
- `horizon`
- `capabilities`
- `metadata`

## Task Identity

`TaskIdentity` identifies the task.

It contains:

- `task_id`
- `version`
- `domain`
- `split`

The identity fields are validated by the task contract.

The split distinguishes task usage such as RL/training from held-out evaluation.

## Environment Binding

`EnvironmentBinding` identifies the environment associated with the task.

It contains:

- `environment_id`
- `environment_version`
- `version`
- `max_episode_steps`

`max_episode_steps` is optional, but when provided it must satisfy the positive-value contract.

## Scenario Specification

`ScenarioSpec` identifies the scenario used for the task.

It contains:

- `scenario_id`
- `generator`
- `configuration`

The scenario configuration is represented as structured mapping data.

## Action Specification

`ActionSpec` defines the action vocabulary and action representation.

It contains:

- `kind`
- `vocabulary`
- `schema`
- `constraints`

The vocabulary must contain valid unique action entries when used by the task.

## Observation Specification

`ObservationSpec` defines the observation representation.

It contains:

- `kind`
- `schema`
- `constraints`

## Reward Specification

`RewardSpec` defines the reward model.

It contains:

- `kind`
- `components`
- `aggregation`

Reward component details must remain compatible with the framework reward contract.

## Termination Specification

`TerminationSpec` defines episode stopping behavior.

It contains:

- `termination_conditions`
- `truncation_conditions`
- `max_episode_steps`

The maximum episode steps value is optional but must be positive when provided.

## Success Specification

`SuccessSpec` defines task success criteria.

It contains:

- `criteria`
- `threshold`
- `conditions`

The task must provide a valid success criterion set.

## Oracle Specification

`OracleSpec` identifies oracle behavior associated with the task.

It contains:

- `oracle_id`
- `implementation`
- `configuration`

Oracle information must not expose held-out evaluation answers to the RL training process.

## Verifier Specification

`VerifierSpec` identifies the verifier associated with the task.

It contains:

- `verifier_id`
- `implementation`
- `configuration`

Verifier execution is part of evaluation and validation rather than agent-side task knowledge.

## Compatibility Specification

`CompatibilitySpec` identifies contract compatibility information.

It contains:

- `contract_version`
- `framework_version`
- `requirements`

Contract versions allow task definitions to be checked against the framework version that executes them.

## Objective

`TaskSpec.objective` is a string describing the task objective.

## Difficulty

`TaskSpec.difficulty` is numeric and must be non-negative.

## Horizon

`TaskSpec.horizon` is either a positive integer or `null`.

## Capabilities

`TaskSpec.capabilities` contains capability identifiers represented as strings.

## Metadata

`TaskSpec.metadata` contains additional structured task metadata.

## Task Authoring Rules

A task author should ensure that:

1. The task identity is complete.
2. The environment binding resolves to a compatible environment.
3. The scenario is reproducible under the seed contract.
4. Actions and observations match the environment spaces.
5. Reward semantics match the reward contract.
6. Termination and truncation semantics are explicit.
7. Success criteria are verifiable.
8. Oracle and verifier definitions are isolated appropriately.
9. Compatibility information is explicit.
10. RL and evaluation task splits are preserved.

## Validation

Task specifications should be validated before registration.

Schema validation checks serialized task representations.

Runtime contract validation checks Python task objects and their semantic constraints.

Both forms should remain aligned.
