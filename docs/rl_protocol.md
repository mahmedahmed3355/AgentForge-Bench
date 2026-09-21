# AgentForge-RL-Bench — RL Protocol

## Purpose

The RL protocol defines how a task, environment, agent, episode, reward system, and trajectory recorder interact during an RL episode.

## Episode Flow

The canonical interaction flow is:

1. Resolve a task.
2. Resolve the task environment.
3. Resolve or generate the scenario.
4. Initialize the episode seed.
5. Reset the environment.
6. Provide the observation to the agent.
7. Receive an action from the agent.
8. Execute the action through the environment.
9. Receive observation, scalar reward, termination flags, and info.
10. Record the transition.
11. Continue until termination or truncation.
12. Persist the resulting trajectory and episode information.

## Task Selection

Tasks are selected through the task catalog/registry layer.

A task has a canonical `TaskSpec` containing its identity, environment binding, scenario, action, observation, reward, termination, success, oracle, verifier, compatibility, and task metadata.

RL and evaluation task splits must remain distinct.

Evaluation tasks are held out from RL training according to the project protocol.

## Scenario

A scenario defines the task instance conditions under which an episode is executed.

Scenario generation and seed management must preserve reproducibility.

## Agent Interaction

The agent receives observations and produces actions.

The environment remains responsible for state transitions and task semantics.

The protocol does not require the environment to expose internal state to the agent.

## Action

Actions must conform to the task action contract and the environment action space.

Invalid actions must not silently alter the intended task semantics.

## Observation

Observations are produced by the environment after reset and after transitions.

They must conform to the task observation contract and declared observation space.

## Reward

The environment returns a scalar reward.

When reward decomposition is required, component-level information is retained separately through the reward contract and/or structured transition information.

The scalar reward remains the value used by the Gymnasium interaction contract.

## Termination

An episode terminates when the task reaches a terminal condition.

The protocol preserves the distinction between:

- semantic termination: `terminated=True`
- truncation: `truncated=True`

## Truncation

Truncation represents an externally or configurationally imposed stopping condition, such as a maximum episode horizon.

A truncated episode is not automatically equivalent to successful task completion.

## Seed

The episode seed is part of the reproducibility contract.

The same task, scenario, seed, and compatible environment configuration should provide reproducible behavior within the deterministic guarantees of the implementation.

## Trajectory

Each executed transition can be recorded into the trajectory system.

A trajectory contains task identity, scenario identity, seed, ordered steps, actions, observations, rewards, reward components, and termination information.

## Long-Horizon Episodes

The benchmark supports long-horizon task interaction.

The protocol must preserve:

- multi-step decision making
- branching behavior
- failure and recovery
- component rewards
- deterministic replay conditions
- trajectory persistence
- episode termination/truncation semantics

## Protocol Boundary

The RL protocol describes interaction semantics.

Task-specific implementation details belong to the task environment and task contract rather than being embedded into the generic protocol.
## Canonical RL API

The canonical RL agent contract is:

`agentforge.agents.rl.base.RLAgent`

The canonical policy contract is:

`agentforge.agents.rl.policy.Policy`

Legacy or integration-facing agent adapters may remain available for compatibility,
but new RL-specific implementations should target the canonical RL contracts.
