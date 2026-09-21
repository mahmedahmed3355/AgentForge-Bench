# AgentForge-RL-Bench — Trajectory Protocol

## Purpose

The trajectory protocol defines the canonical representation of an executed episode and its ordered transitions.

## Canonical Trajectory

The canonical trajectory model is `agentforge.runtime.trajectory.Trajectory`.

A trajectory contains:

- `episode_id`
- `task_id`
- `scenario_id`
- `seed`
- `steps`

The steps are ordered and represented by `TrajectoryStep`.

## TrajectoryStep

The canonical `TrajectoryStep` contains:

- `episode_id`
- `task_id`
- `scenario_id`
- `seed`
- `step_index`
- `observation`
- `action`
- `reward_total`
- `reward_components`
- `terminated`
- `truncated`
- `info`

## Step Ordering

`step_index` identifies the position of a transition within the episode.

Trajectory steps must preserve execution order.

## Reward Representation

`reward_total` is the scalar transition reward.

`reward_components` contains the structured reward breakdown associated with the transition.

The component representation must remain compatible with the canonical reward contract.

## Termination Information

Each step records:

- `terminated`
- `truncated`

The two values preserve the Gymnasium distinction between semantic termination and truncation.

## Transition Information

`observation` records the observation associated with the step.

`action` records the action taken by the agent.

`info` carries structured auxiliary information.

## Episode Identity

Every step is associated with:

- episode ID
- task ID
- scenario ID
- seed

This allows trajectory records to be associated with the exact execution context.

## Persistence

The trajectory recorder is responsible for recording executed steps and supporting persistence/export behavior.

Persisted trajectories should retain enough information to reconstruct the episode execution context and support evaluation analysis.

## Analysis

Trajectory data supports analysis of:

- decisions
- branches
- failures
- recovery
- reward components
- termination
- truncation
- capability evidence

## Compatibility

The framework contains compatibility-oriented trajectory recorder behavior.

The canonical trajectory data model is defined by `runtime/trajectory.py`.

Any compatibility representation must not silently change the canonical trajectory semantics.


## Trajectory Recorder

The runtime trajectory recording component is `agentforge.runtime.trajectory_recorder.TrajectoryRecorder`.

`TrajectoryRecorder` captures episode interaction data and produces trajectory steps using the canonical `agentforge.runtime.trajectory.TrajectoryStep` contract.

The recorder does not define a second canonical `TrajectoryStep` model. It uses the canonical trajectory model from `runtime/trajectory.py`.
