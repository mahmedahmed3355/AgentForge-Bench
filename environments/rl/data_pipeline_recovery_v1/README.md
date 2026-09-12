# Long-Horizon Data Pipeline Recovery V1

AgentForge-Bench RL environment for long-horizon recovery of stateful data
pipelines.

## Overview

The environment models a pipeline:

Input → Schema Validation → Cleaning → Transformation →
Aggregation → Output Validation → Output

The agent encounters an incident and must inspect, diagnose, select a recovery
branch, apply changes, execute the pipeline, respond to downstream
consequences, recover or replan when necessary, and complete global validation.

## Why This Task Exists

Task 002 is designed to extend the long-horizon reasoning requirements of
AgentForge-Bench.

Compared with a short single-step repair task, this environment introduces:

- sequential dependencies
- multiple incident families
- branch-specific recovery
- nested recovery strategies
- downstream consequences
- delayed failure detection
- recovery
- replanning
- global validation

## Public Execution Model

The public environment exposes mechanics required for interaction.

The causal dependency graph is:

```text
Input
  ↓
Schema Validation
  ↓
Cleaning
  ↓
Transformation
  ↓
Aggregation
  ↓
Output Validation
  ↓
Output
A component can be locally repaired while downstream components remain
invalid or blocked.

Actions

The current action contract includes:

inspect_pipeline
inspect_component
analyze_failure
select_branch
select_strategy
modify_component
modify_config
run_pipeline
check_schema
check_output
recover
replan
final_verify
Branch Families

The public branch model contains:

schema recovery
transformation recovery
data quality recovery
resource recovery

Each family provides multiple recovery strategies.

Statefulness

The environment tracks:

logical progress
incident state
diagnosis
selected branch
selected strategy
component validity
pipeline execution
failure detection
recovery
replanning
downstream inconsistency
global validation
terminal success
Anti-Cheat

The agent-facing package does not intentionally contain:

oracle answers
hidden answers
reference solutions
hidden evaluation truth
expected optimal trajectories

See:

docs/ANTI_CHEAT.md
docs/ANTI_MEMORIZATION.md
Evaluation Boundary

Public mechanics live in the task package.

Hidden scenario data and evaluator-only truth belong under the evaluator boundary
and must not be imported by the agent-facing environment.

Current Status

Development.

The execution engine and causal layer are under iterative validation.

This task must not be considered frozen until the complete release checklist
has passed.
