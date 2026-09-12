# AgentForge-Bench Architecture

AgentForge-Bench is designed as a modular benchmark framework for training and evaluating long-horizon AI agents.

## Core Flow

Taskset
  |
  +--> TaskData
  |
  +--> Task
  |      +--> State
  |      +--> Actions
  |      +--> Tools
  |      +--> Scoring
  |      +--> Stop Conditions
  |
  v
Harness
  |
  v
Runtime
  |
  v
Trace
  |
  v
Evaluation / RL Training

## Repository Layout

agentforge/
    Core framework.

environments/
    Prime V1-compatible benchmark environments.

tests/
    Global framework and environment tests.

docs/
    Architecture, methodology, authoring, and evaluation documentation.

datasets/
    Benchmark datasets and evaluation data.

## Design Principles

- Reproducibility
- Deterministic environment initialization
- Stateful long-horizon interaction
- Explicit task contracts
- Verifiable outcomes
- Hidden evaluation
- Truth isolation
- Anti-cheating design
- Anti-memorization
- Reward-hacking resistance
- Failure and recovery
- Prime V1 compatibility
- Training/evaluation separation
