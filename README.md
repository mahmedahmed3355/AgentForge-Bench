# AgentForge-RL-Bench

AgentForge-RL-Bench is a long-horizon reinforcement-learning benchmark framework for training and evaluating AI agents inside task-specific environments.

## Architecture

The framework provides:

- Agent adapter contract
- Per-task environment contract
- Long-horizon task execution
- Branching and decision points
- Failure and recovery paths
- Multi-component rewards
- Step-level trajectory persistence
- Checkpointing
- Trajectory analysis
- Capability reports
- Deterministic evaluation contexts
- Multi-agent comparison
- Repeated multi-seed evaluation
- Batch capability analysis
- Task authoring and validation
- Task registry and catalog
- Task manifests and integrity metadata

## Planned Task Library

The benchmark task library is being built across three engineering domains:

1. CUDA / GPU engineering
2. Distributed training
3. Backend / infrastructure engineering

The target release contains 100 long-horizon RL tasks.

Each task owns its environment and task metadata.

## Installation

From the repository:

    pip install .

For development:

    pip install -e ".[dev]"

## Development

Run the complete test suite:

    pytest -q

Compile the package:

    python -m compileall -q src

## Task Model

Each RL task is designed around a long interaction horizon with:

- observation
- decision making
- branching
- action execution
- validation
- failure detection
- recovery
- incremental progress
- terminal success/failure

The canonical RL contract requires a minimum horizon of 100 steps.

## Evaluation

An installed benchmark package is intended to allow a client to:

1. Select an agent implementation.
2. Select one or more benchmark tasks.
3. Execute training or evaluation episodes.
4. Persist trajectories.
5. Analyze agent capability.
6. Compare multiple agents under the same evaluation context.
7. Export machine-readable evaluation reports.

## License

MIT
