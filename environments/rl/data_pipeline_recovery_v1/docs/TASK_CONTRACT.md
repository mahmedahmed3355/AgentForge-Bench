# Task 002 — Long-Horizon Data Pipeline Recovery V1

## Status

Development

## Difficulty Target

Medium

Task 002 is intentionally harder than the first CUDA optimization task through
longer causal dependencies, branch-specific recovery, nested recovery, and
replanning.

## Core Pipeline

Input → Schema Validation → Cleaning → Transformation →
Aggregation → Output Validation → Output

## Incident Families

1. Schema failure
2. Transformation failure
3. Data quality failure
4. Resource / execution failure

## Agent Responsibilities

The agent must:

1. inspect the pipeline
2. diagnose the incident
3. select a valid recovery branch
4. select a valid strategy
5. modify the affected system
6. execute the pipeline
7. inspect resulting consequences
8. recover when required
9. replan when required
10. validate the global output
11. reach final verification

## Long-Horizon Requirement

The successful trajectory should require meaningful sequential decisions.

The environment must not reduce the task to a fixed action-count puzzle.

## Branching

Each incident family has multiple strategies.

Some strategies create a nested replanning path.

## Causal Requirement

Local component validity is not equivalent to global pipeline validity.

Downstream consequences must be observable through execution and validation.

## Anti-Cheat

Hidden evaluation truth is isolated from the public package.

## Anti-Memorization

Evaluation should include unseen scenarios and varying causal consequences.

## Release Requirements

Before freezing:

- unit tests
- execution tests
- causal tests
- reward-hacking tests
- anti-cheat audit
- anti-memorization audit
- deterministic reset
- Gymnasium compatibility
- Prime V1 compatibility
- standalone installation
- clean-room execution
- Oracle validation
- independent verifier validation
- hidden evaluation
- trajectory reporting
