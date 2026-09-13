# Branching Decision & Recovery V1

AgentForge-Bench Task 003.

A long-horizon interactive RL environment for decision-making under partial observability, information acquisition, branching, delayed consequences, recovery, and replanning.

## Research Identity

Task 003 is intentionally different from Task 001 and Task 002.

- Task 001 focuses on CUDA kernel optimization, experimentation, and performance-driven decisions.
- Task 002 focuses on causal debugging and long-horizon data pipeline recovery.
- Task 003 focuses on decision-making under uncertainty, information acquisition, plausible branching, delayed consequences, temporal credit assignment, recovery, and replanning.

The benchmark is designed to study whether an agent can gather useful information, form hypotheses, commit to costly decisions, recognize delayed consequences, recover without resetting history, and replan toward a globally verified terminal state.

## Core Loop

Partial Observation
→ Information Gathering
→ Hypothesis
→ Branch Decision
→ Costly Action
→ Delayed Consequence
→ New Observation
→ Recovery / Replanning
→ Global Validation

## Primary Capabilities

The environment evaluates:

- partial observability
- information acquisition
- hypothesis formation
- branch selection
- strategy selection
- costly decisions
- risk-aware planning
- delayed consequences
- failure diagnosis
- recovery without episode reset
- replanning after new information
- temporal and causal credit assignment
- sparse and progress-based reward
- terminal global validation
- trajectory-level reporting
- unseen scenario generalization

## Environment Components

Task 003 contains the following major components:

- state model
- observation model
- information acquisition
- inspection adapter
- decision graph
- branch semantics
- strategy semantics
- cost and risk model
- hidden dependencies
- delayed consequence model
- failure and cascade model
- recovery model
- replanning model
- reward contract
- terminal conditions
- Oracle
- independent Verifier
- hidden scenario evaluation
- anti-cheat boundary
- anti-memorization boundary
- trajectory recording
- AgentReport
- Gymnasium adapter
- Prime Verifiers V1 integration
- standalone package installation
- clean-room validation support
- agent calibration support

## Decision Model

The environment contains multiple plausible decision branches.

The agent must select a branch and compatible strategy after gathering sufficient information.

Branches and strategies have different:

- costs
- risks
- information requirements
- downstream consequences
- recovery behavior

A locally successful action is not necessarily globally successful.

Some branches deliberately produce delayed consequences so that an agent must continue interacting with the environment before the true outcome becomes observable.

## Information Acquisition

Information-gathering actions expose partial observations about the system.

Examples include:

- system inspection
- component inspection
- dependency inspection
- history inspection
- state probing
- metrics inspection
- validation queries

Repeated information requests do not provide unlimited progress or reward.

The observation boundary does not expose the correct branch, hidden root cause, optimal strategy, oracle trajectory, or reference solution.

## Delayed Consequences

Some costly decisions schedule consequences that become observable only after additional interaction.

This creates a temporal dependency between:

1. an earlier decision
2. subsequent execution
3. a delayed failure
4. new evidence
5. recovery
6. replanning
7. final validation

The agent must therefore reason over a state trajectory rather than treating each action independently.

## Recovery and Replanning

Recovery does not reset the episode or erase historical decisions.

When a delayed failure is detected, the agent may need to:

- recover the affected system
- revise its hypothesis
- replan
- select a different branch
- select a compatible strategy
- execute the revised plan
- globally validate the final state

Successful recovery restores the active causal state while preserving the historical trajectory.

## Reward

Reward is progress-oriented.

Reward can reflect:

- useful information acquisition
- hypothesis formation
- decision progress
- repair or recovery progress
- replanning progress
- successful terminal verification

Repeated inspection, redundant behavior, invalid actions, and ineffective actions are not intended to provide a reliable reward-farming path.

Terminal success is dominated by actual final-state correctness and global validation.

## Agent Report

Task 003 provides an `AgentReport` model for trajectory-level analysis.

The report records:

- task_id
- status
- planning_depth
- information_queries
- branch_decisions
- wrong_decisions
- delayed_failures
- recoveries
- replans
- redundant_actions
- invalid_actions
- total_cost
- final_reward
- terminal
- success

Reports can be serialized with `to_dict()`.

The report is intended for analyzing agent behavior and failure modes, not only pass/fail outcomes.

## Oracle and Verifier

The Oracle provides a reference mechanism for validating that the environment contains a solvable successful trajectory.

The Oracle is not exposed as agent-facing truth.

The Verifier independently evaluates the resulting terminal state.

Verification requires the relevant terminal conditions, including:

- terminal state
- successful final state
- valid branch and strategy
- no unresolved active failure
- no pending recovery
- no pending replanning
- global validation
- sufficient logical progress

The verifier does not rely on the agent's reported result.

## Hidden Evaluation

Public scenarios are stored in:

`data/scenarios.json`

Hidden scenarios are separated from the agent-facing package and are generated/evaluated through the evaluator layer.

Hidden evaluation varies scenario properties such as:

- scenario family
- topology
- dependency structure
- symptom profile
- branch costs
- branch risks
- strategy costs
- strategy risks
- delayed consequence timing

The hidden scenarios are intended to test behavior on unseen configurations rather than memorization of public examples.

## Public Scenario Families

The current public scenarios cover:

- dependency
- configuration
- cascade

The scenario generator supports additional variation for hidden evaluation.

## Prime Verifiers V1

Task 003 integrates with Prime Verifiers V1.

The Prime implementation provides:

- TaskData
- Task
- Taskset
- Taskset configuration
- Prime State
- Prime Toolset
- Prime Environment
- environment/taskset loading

The package declares the Prime evaluation configuration in `pyproject.toml`.

## Gymnasium

Task 003 provides a real Gymnasium environment adapter.

The adapter exposes:

- structured observation space
- discrete action space
- reset
- step
- deterministic seeded execution

The native environment remains responsible for the benchmark semantics.

## Installation

Install the task as a standalone package from this directory:

`environments/rl/branching_decision_recovery_v1`

The task declares its runtime dependencies in `pyproject.toml`.

The package is designed to operate without requiring the AgentForge-Bench repository itself as a runtime dependency.

## Testing

Run the complete Task 003 regression suite from the repository root:

`pytest -q environments/rl/branching_decision_recovery_v1`

The suite covers:

- task structure
- imports
- decision graph
- observation contract
- Gymnasium validation
- reward behavior
- hidden scenarios
- Oracle behavior
- Verifier behavior
- anti-cheat boundaries
- scenario generation
- agent reporting
- final audit behavior

## Difficulty

Initial target difficulty:

Hard

The architecture supports later Hard → Very Hard scaling through meaningful increases in:

- planning depth
- information dependencies
- branch ambiguity
- causal dependencies
- delayed consequences
- recovery complexity
- unseen scenario variation

Difficulty must come from meaningful state dependencies, information dependencies, planning depth, branching, delayed consequences, and recovery.

Do not add arbitrary failures or unnecessary actions merely to increase difficulty.

## Benchmark Status

Current status:

Release Candidate

Task 003 has passed the current regression and final report audit.

The environment is not considered permanently frozen until release-candidate validation, standalone clean-room validation, Prime validation, and agent calibration are completed.

## Version

Current task package version:

`0.1.0`

The version is kept separate from the benchmark task identity and may be incremented when the release candidate is finalized.

## Task Identity

Task ID:

`branching-decision-recovery-v1`

Task number:

`003`

Domain:

Decision Making / Stateful Systems / Agentic RL / Recovery / Replanning

Primary research question:

Can an agent make reliable long-horizon decisions under partial observability when information has a cost, branches have different delayed consequences, and successful completion requires recovery and replanning rather than resetting the episode?

## Design Principle

The benchmark should reward agents for making meaningful progress through the underlying state dependencies.

Difficulty should emerge from reasoning-relevant interactions between:

- information
- hypotheses
- decisions
- costs
- risks
- dependencies
- delayed consequences
- failures
- recovery
- replanning
- final validation

The benchmark should not rely on arbitrary action inflation or artificial failure injection.
