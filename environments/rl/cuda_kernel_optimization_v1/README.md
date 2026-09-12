# CUDA Kernel Optimization V1

AgentForge-Bench reinforcement-learning environment for long-horizon CUDA kernel optimization.

## Objective

The agent must optimize a CUDA kernel while preserving correctness and improving measured performance.

## Horizon

The task contains 20 logical stages.

The logical stages are not a fixed number of agent actions.

An episode may require additional actions for:

- inspection
- diagnosis
- compilation
- correctness checks
- benchmarking
- analysis
- recovery
- re-validation

## Canonical Workflow

1. Inspect
2. Identify optimization target
3. Inspect implementation
4. Modify candidate
5. Compile
6. Check correctness
7. Benchmark
8. Analyze results
9. Modify again
10. Compile
11. Check correctness
12. Benchmark
13. Analyze
14. Detect regression
15. Recover
16. Re-validate correctness
17. Benchmark again
18. Confirm performance
19. Prepare final candidate
20. Final verification

## Reward

Reward components include:

- progress
- milestones
- recovery
- correctness
- performance
- efficiency
- final success

Final success is intentionally dominant.

## Evaluation

The environment separates scenario categories:

- train
- eval
- hidden

The hidden scenario is not intended to be exposed to training logic.

## Current Implementation Status

This first implementation establishes the Prime V1 taskset and the benchmark contracts.

The actual CUDA compilation, runtime benchmarking, randomized correctness testing, and sandboxed kernel execution are added only after the taskset contract passes baseline validation.

This prevents CUDA infrastructure failures from being confused with task-design failures.
