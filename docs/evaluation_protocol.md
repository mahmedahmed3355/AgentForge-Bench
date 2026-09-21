# AgentForge-RL-Bench — Evaluation Protocol

## Purpose

The evaluation protocol defines how held-out tasks are executed, verified, and reported.

Evaluation is separate from RL training.

## Evaluation Isolation

Evaluation tasks are held out from the RL training split.

The evaluation process must not expose evaluation answers or verifier implementation details to the training process.

Task IDs, versions, seeds, environment versions, and evaluation results should remain traceable.

## Evaluation Runner

The framework contains evaluation runner functionality responsible for executing evaluation episodes.

The public evaluation layer resolves evaluation tasks through the task catalog and keeps evaluation task selection separate from RL task selection.

The runtime evaluation layer executes action sequences against an environment and produces episode results.

## EpisodeResult

An evaluation episode produces an `EpisodeResult`.

The canonical fields are:

- `observation`
- `total_reward`
- `step_count`
- `terminated`
- `truncated`
- `trajectory`

`mean_reward` is available on the aggregate evaluation result.

## Episode Execution

An evaluation episode follows the environment protocol:

1. Resolve the task.
2. Resolve the environment.
3. Execute the supplied agent/action behavior.
4. Record observations, actions, rewards, and termination information.
5. Produce an `EpisodeResult`.
6. Construct verification context.
7. Execute the task verifier.
8. Preserve verification diagnostics and result metadata.

## Verifier Integration

The evaluation layer connects the task verifier to the executed episode.

The verifier execution flow includes:

1. Resolve the verifier from the task specification.
2. Construct `VerifierContext`.
3. Validate episode-level conditions.
4. Validate trajectory-level conditions.
5. Validate reward conditions.
6. Validate success conditions.
7. Diagnose failures when applicable.
8. Execute the canonical verifier.
9. Produce `VerificationResult`.

## VerificationResult

`VerificationResult` records structured verification output.

It includes:

- `verified`
- `verifier_id`
- `task_id`
- `episode`
- `trajectory`
- `reward`
- `success`
- `diagnostics`
- `metadata`

The `passed` compatibility property reflects the `verified` value.

## Held-Out Evaluation

Evaluation results must identify the evaluation task and execution context.

A valid evaluation record should make it possible to determine:

- task ID
- task version
- environment version
- scenario
- seed
- agent/model configuration
- checkpoint or policy state when applicable
- episode result
- verification result

## Evaluation Integrity

The benchmark evaluation process must preserve:

- training/evaluation separation
- verifier isolation
- reproducibility information
- task identity
- environment identity
- trajectory evidence
- verification evidence

## Reporting Boundary

Evaluation produces machine-readable results that can be consumed by reporting and comparison layers.

Reporting should describe what the agent did, what was verified, and what remains missing without changing the underlying evaluation result.

## EvaluationRunner

The canonical runtime evaluation component is `EvaluationRunner`.

Its execution boundary is responsible for running evaluation action sequences against the resolved environment and producing `EvaluationResult` / `EpisodeResult` data for verifier processing.

The public evaluation layer separately handles held-out evaluation task selection through the task catalog.


## Evaluation Runtime Component

The canonical runtime evaluation component is `agentforge.runtime.evaluation_runner.EvaluationRunner`.

`EvaluationRunner` executes evaluation episodes and produces `EvaluationResult` data containing episode results, aggregate reward information, and verification results where applicable.

The evaluation protocol therefore uses the runtime `EvaluationRunner` implementation as the execution component for the documented evaluation lifecycle.


## Verifier Execution

The verifier contract is represented by `BaseVerifier`, which defines the verifier interface used by the framework.

`VerifierExecutor` is the executable verifier component responsible for invoking registered verifier implementations and producing verification results within the evaluation flow.
## Canonical Evaluation API

The canonical evaluation runtime component is:

`agentforge.runtime.evaluation_runner.EvaluationRunner`

Legacy integration or orchestration modules may remain as compatibility layers,
but the runtime evaluation contract is defined by `EvaluationRunner`.
