
Task 002 Evaluator Boundary

This directory is reserved for evaluator-only logic.

Isolation Rule

Evaluator-only information must not be imported by the agent-facing package.

The evaluator may contain:

hidden scenarios
scenario-specific evaluation predicates
hidden perturbations
independent verification logic
benchmark-only metadata

The public environment must remain usable without access to this directory's
hidden truth.

Anti-Cheat Requirement

Do not place reference trajectories, oracle answers, expected optimal actions,
or secret scenario labels in files distributed to the agent.

Evaluation Architecture

The intended boundary is:

Agent
  ↓
Public Environment
  ↓
Observable State / Consequences
  ↓
Independent Evaluator
  ↓
Hidden Evaluation Data

The evaluator should verify outcomes independently rather than trusting
agent-reported success.
