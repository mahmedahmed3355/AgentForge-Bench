# Task 002 — Anti-Cheat and Truth Isolation

## Purpose

Task 002 is designed so that the agent must reason over pipeline state and
execution consequences rather than retrieve a reference answer.

## Agent-Facing Package Rules

The package must not contain:

- oracle answers
- hidden answers
- reference solutions
- hidden scenario truth
- expected optimal action sequences
- secret evaluation labels
- evaluator-only scenario parameters
- benchmark answer keys

Public code may contain:

- task mechanics
- action contracts
- state contracts
- causal dependency mechanics
- public incident families
- public scenario identifiers
- public validation behavior
- reward mechanics
- interface documentation

## Truth Isolation

Scenario-specific evaluation truth belongs under the evaluator boundary.

The public environment must not import evaluator-only scenario data.

The evaluator may contain hidden scenarios, hidden predicates, and evaluation
metadata that are not distributed with the agent-facing task package.

## No Oracle Leakage

Oracle logic must not be encoded in:

- observations
- action names
- tool descriptions
- public scenario files
- README examples
- reward messages
- error strings
- package constants

## Runtime Leakage

Public `info` should describe observable execution consequences only.

It must not expose:

- the correct future action
- the optimal recovery strategy
- hidden labels
- evaluator decisions
- hidden scenario identifiers
- reference trajectory information

## Structural Audit

Before release, scan the agent-facing package for forbidden markers such as:

`oracle_answer`

`hidden_answer`

`reference_solution`

`reference_answer`

`ground_truth`

`hidden_solution`

`secret_solution`

`expected_optimal`

A marker audit is necessary but not sufficient. Semantic review of
agent-visible behavior is also required.
