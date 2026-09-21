# AgentForge-RL-Bench — Environment Contract

## Purpose

The environment contract defines the canonical interface between the benchmark runtime and an RL task environment.

The framework uses Gymnasium-compatible environment semantics for RL interaction.

## Canonical Environment Lifecycle

An environment represents one task instance and owns the state transition process for an episode.

The lifecycle is:

1. Create or resolve the environment for a task.
2. Reset the environment at the beginning of an episode.
3. Observe the initial observation and reset information.
4. Execute actions through `step(action)`.
5. Produce the next observation, scalar reward, termination flags, and info.
6. Continue until `terminated` or `truncated`.
7. Close the environment when the episode/runtime lifecycle is complete.

## Reset Contract

The Gymnasium-facing reset operation is:

    reset(*, seed=None, options=None)
        -> (observation, info)

Reset establishes the initial episode state.

The seed belongs to the episode initialization contract and must be handled deterministically by the environment implementation.

## Step Contract

The canonical Gymnasium step shape is:

    step(action)
        -> (observation, reward, terminated, truncated, info)

The five returned values have distinct meanings:

- `observation`: observation available to the agent after the transition.
- `reward`: scalar reward for the transition.
- `terminated`: the task reached a terminal state according to task semantics.
- `truncated`: the episode ended because of a truncation condition such as a configured horizon.
- `info`: structured auxiliary information.

`terminated` and `truncated` must not be collapsed into one flag.

## Action Space

Every Gymnasium-compatible RL environment exposes `action_space`.

The action space describes the valid action representation expected by the environment.

Invalid actions should be rejected according to the framework action/error contract rather than silently changing task semantics.

## Observation Space

Every Gymnasium-compatible RL environment exposes `observation_space`.

The observation returned by `reset()` and `step()` must conform to the declared observation representation.

## Reward

The environment returns a scalar reward through the Gymnasium step contract.

When reward decomposition is required, detailed reward information belongs in the structured `info` payload while the Gymnasium reward remains scalar.

The framework reward contracts define component-level reward representation and validation.

## Info Contract

The `info` value carries structured auxiliary information associated with the transition.

Reward components and other diagnostic information may be represented here without changing the scalar Gymnasium reward contract.

## Determinism and Seed

Environment implementations must honor the framework seed contract.

A seed must determine the episode initialization and any environment-owned stochastic behavior covered by the task contract.

## Termination and Truncation

Termination represents task-semantic completion or failure.

Truncation represents an external or configured stopping condition, such as a maximum episode horizon.

The two conditions are represented independently.

## Canonical Contract Summary

| Component | Contract |
|---|---|
| `reset` | `reset(*, seed=None, options=None) -> (observation, info)` |
| `step` | `step(action) -> (observation, reward, terminated, truncated, info)` |
| `action_space` | Required |
| `observation_space` | Required |
| reward | Scalar |
| reward details | `info` / reward contract |
| termination | `terminated` |
| truncation | `truncated` |
| auxiliary data | `info` |
| seeding | Deterministic seed contract |

## Compatibility

The canonical Gymnasium-facing implementation is provided by the environment layer and its adapter.

Task-specific environments must preserve the framework lifecycle, reward, termination, truncation, and seed semantics.
