# AgentForge-Bench

AgentForge-Bench is a benchmark and environment framework for training and evaluating long-horizon AI agents and reinforcement-learning systems.

The project focuses on realistic stateful environments where agents must reason, act, observe consequences, recover from failures, and complete multi-stage objectives.

## Architecture

The benchmark follows a Prime Verifiers V1-oriented architecture:

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
RL Training / Evaluation

## Repository Structure

agentforge/
    Core AgentForge framework.

environments/
    Benchmark environments and Prime V1 tasksets.

evaluator/
    Evaluation boundary, hidden scenarios, and evaluation harnesses.

benchmarks/
    Benchmark registry, split metadata, and benchmark schemas.

tests/
    Automated unit, integration, and environment tests.

docs/
    Architecture, benchmark methodology, and authoring documentation.

datasets/
    Training and evaluation datasets.


## Benchmark Design

AgentForge-Bench targets:

- Long-horizon reasoning
- Stateful interaction
- Branching decisions
- Failure detection
- Recovery and replanning
- Delayed consequences
- Unseen-scenario generalization
- Tool interaction
- CUDA/GPU engineering
- Data pipeline recovery
- Agentic coding and engineering workflows
- Robust evaluation

## Current Benchmark Status

AgentForge-Bench currently contains its first frozen benchmark environment:

- Task 001 — CUDA Kernel Optimization V1
- Long-horizon stateful interaction
- Branching optimization decisions
- Failure detection and recovery
- Oracle and independent verifier
- Hidden evaluation boundary
- Anti-cheating and anti-memorization checks
- Reward-hacking resistance
- Gymnasium adapter
- Prime Verifiers V1 environment integration
- Standalone package validation
- Clean-room installation and runtime validation

Task 001 is frozen as a reference environment. New benchmark tasks are
added incrementally under `environments/`.

The benchmark registry is maintained in `benchmarks/registry.yaml`.

## Evaluation Principles

Tasks are designed around:

- Oracle correctness
- Independent verification
- Hidden evaluation
- Truth isolation
- Anti-cheating mechanisms
- Anti-memorization
- Reward-hacking resistance
- Deterministic and reproducible execution
- Failure and recovery semantics
- Train/evaluation/hidden scenario separation

## Prime V1

AgentForge-Bench is designed to integrate with Prime Verifiers V1 and its Taskset / Task / Harness / Runtime architecture.

Prime compatibility is treated as an end-to-end property. A task is not considered production-ready merely because the package imports successfully.

Validation includes:

1. Installation
2. Taskset loading
3. Task construction
4. Environment execution
5. Oracle validation
6. Verifier validation
7. Hidden evaluation
8. Trace generation
9. Reproducibility
10. RL training smoke testing

## Development Status

Current status: Foundation initialized.

Canonical benchmark tasks will be added incrementally after the framework foundation passes its baseline checks.

## License

License will be finalized before the first public benchmark release.
