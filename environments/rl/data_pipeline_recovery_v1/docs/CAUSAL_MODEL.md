# Task 002 — Causal Pipeline Model

## Pipeline

The public execution model contains seven components:

Input → Schema Validation → Cleaning → Transformation →
Aggregation → Output Validation → Output

## Dependency Principle

Every downstream component depends on the validity of its upstream
dependencies.

A local repair therefore does not automatically imply global correctness.

## Execution

Pipeline execution evaluates the current component state.

The execution result can expose:

- whether execution completed
- invalid components
- blocked components
- downstream consequences

The agent must use these observable consequences to decide whether additional
recovery or replanning is necessary.

## Recovery

Recovery changes component state but does not automatically certify the final
output.

Final correctness requires global validation.

## Hidden Evaluation Boundary

The causal dependency mechanics are public.

Scenario-specific hidden conditions, hidden evaluation data, and evaluator-only
truth remain outside the agent-facing package.
