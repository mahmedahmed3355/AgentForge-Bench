# AgentForge-RL-Bench — Architecture

## Purpose

AgentForge-RL-Bench is organized as a layered benchmark framework for long-horizon RL task execution, trajectory recording, evaluation, verification, and reporting.

## High-Level Flow

The primary execution flow is:

    Task Registry
        |
        v
    Task / Scenario
        |
        v
    Environment
        |
        v
    Agent Interaction
        |
        v
    Trajectory
        |
        +------> Reward / Episode Analysis
        |
        v
    Evaluation
        |
        v
    Verifier
        |
        v
    Results / Reporting

## Task Contract Layer

The task contract defines the canonical task specification.

Primary model:

`agentforge.contracts.task.TaskSpec`

The task specification binds:

- identity
- environment
- scenario
- actions
- observations
- reward
- termination
- success
- oracle
- verifier
- compatibility

## Task Registry

The task registry resolves registered task definitions.

The canonical runtime task registry is responsible for task registration, replacement, removal, lookup, and access to the registered task components.

Task selection and dataset APIs operate above the registry.

## Environment Layer

The environment layer provides task execution semantics.

Important components include:

- `AgentForgeEnv`
- `NativeEnvironment`
- `GymnasiumAdapter`
- `EnvironmentState`
- `Transition`

The Gymnasium adapter provides the Gymnasium-compatible interaction boundary.

## Scenario Layer

The scenario layer represents the task-instance conditions used during an episode.

Scenario generation is associated with the task specification and must preserve seed/reproducibility semantics.

## Runtime Layer

The runtime layer coordinates execution.

Important components include:

- task catalog
- environment factory
- episode runner
- evaluation runner
- trajectory recorder
- reward engine
- run registry
- reporting-related runtime components

## Reward Layer

The canonical reward implementation is located under:

`agentforge.runtime.contracts.rewards`

It provides:

- `RewardComponent`
- `RewardBreakdown`
- `RewardEngine`

The reward engine validates component data and weighted reward aggregation.

## Trajectory Layer

The canonical trajectory data model is:

`agentforge.runtime.trajectory`

It provides:

- `Trajectory`
- `TrajectoryStep`

The trajectory recorder provides recording and persistence behavior around executed transitions.

## Evaluation Layer

Evaluation executes held-out evaluation tasks separately from RL training.

Evaluation produces episode results and invokes verifier execution for structured validation.

## Verifier Layer

The verifier layer provides:

- verifier contracts
- verifier registry
- verifier executor
- episode verification
- trajectory verification
- reward verification
- success verification
- failure diagnostics

Verifier execution is isolated from the agent's task-solving interface.

## Reporting Layer

Reporting consumes experiment, trajectory, evaluation, and verification evidence.

Reports should preserve the distinction between:

- observed agent behavior
- reward results
- verification results
- missing or failed criteria

Reporting does not redefine the underlying task or evaluation contract.

## Data and Integrity Boundaries

The architecture preserves several boundaries:

### RL vs Evaluation

RL training tasks and held-out evaluation tasks remain separate.

### Agent vs Verifier

The verifier is an evaluation-side component and should not expose hidden evaluation information to the agent.

### Task vs Environment

Task specification describes what the task requires.

The environment implements how the task state evolves.

### Environment vs Runtime

The environment owns state transitions.

The runtime owns orchestration, recording, persistence, and higher-level execution flow.

## Contract Stability

Before a framework contract is considered frozen:

1. Canonical interfaces must be defined.
2. Runtime implementations must match the interfaces.
3. Tests must validate the interfaces.
4. Serialized schemas must match the runtime contracts.
5. Documentation must describe the actual contract.
6. Compatibility behavior must be explicitly identified.

## Current Compatibility Note

The repository contains compatibility-oriented components from earlier framework layers.

When a canonical contract exists, new task implementations should target the canonical contract rather than introducing another parallel contract.


## Verifier Execution Layer

The verifier layer includes `BaseVerifier` as the verifier contract and `VerifierExecutor` as the runtime execution component.

`VerifierExecutor` operates on registered verifier implementations and integrates verifier execution with the evaluation and reporting flow.
## Canonical API Policy

The framework maintains one canonical public API per core concept.

### Canonical APIs

| Concept | Canonical API |
|---|---|
| Task specification | `agentforge.contracts.task.TaskSpec` |
| Environment | `agentforge.environment.base.AgentForgeEnv` |
| Gymnasium integration | `agentforge.environment.gym_adapter.GymnasiumAdapter` |
| Environment state | `agentforge.environment.state.EnvironmentState` |
| Transition | `agentforge.environment.transition.Transition` |
| Task registry | `agentforge.runtime.task_registry.TaskRegistry` |
| Task catalog | `agentforge.runtime.catalog.TaskCatalog` |
| Episode execution | `agentforge.runtime.episode_runner.EpisodeRunner` |
| Evaluation | `agentforge.runtime.evaluation_runner.EvaluationRunner` |
| Trajectory | `agentforge.runtime.trajectory.Trajectory` |
| Trajectory recording | `agentforge.runtime.trajectory_recorder.TrajectoryRecorder` |
| Reward model | `agentforge.runtime.contracts.rewards.RewardEngine` |
| Verifier contract | `agentforge.verifier.contracts.BaseVerifier` |
| Verifier execution | `agentforge.verifier.executable.VerifierExecutor` |
| RL agent | `agentforge.agents.rl.base.RLAgent` |
| RL policy | `agentforge.agents.rl.policy.Policy` |

### Legacy Modules

Modules outside the canonical paths above may remain for compatibility, integration,
reporting, orchestration, oracle, or adapter responsibilities.

A legacy module must not introduce a second canonical implementation of the same
core contract. Compatibility modules should delegate to or re-export the canonical
implementation where applicable.

The canonical API is the source of truth for new framework code.


## Legacy Module Policy

Modules listed below are compatibility modules and are not canonical APIs. New framework code should depend on the canonical APIs documented above. Legacy modules remain available only to preserve compatibility with existing integrations.

### Legacy Compatibility Modules

- `src/agentforge/adapter/agent.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/adapter/gymnasium.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/adapter/agentforge_bench.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/integration/episode.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/integration/evaluation.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/orchestration/episode.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/orchestration/evaluation.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/oracle/trajectory.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/reporting/episode_summary.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/reporting/trajectory_analysis.py` — compatibility-only; do not introduce new dependencies on this module.
- `src/agentforge/reporting/verifier_results.py` — compatibility-only; do not introduce new dependencies on this module.

Canonical API ownership remains with the documented `agentforge.contracts`, `agentforge.environment`, `agentforge.runtime`, `agentforge.verifier`, and `agentforge.agents.rl` modules.
