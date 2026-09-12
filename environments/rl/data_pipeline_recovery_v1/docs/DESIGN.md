# Long-Horizon Data Pipeline Recovery V1

## Objective

The agent must recover a stateful data pipeline after an incident.

The task is intentionally designed around diagnosis, branching repair
strategies, cascading consequences, recovery, replanning, and final
global validation.

## Pipeline

Input
  -> Schema Validation
  -> Cleaning
  -> Transformation
  -> Aggregation
  -> Output Validation
  -> Output

A local repair is not sufficient for success.

The agent must establish that the complete pipeline is globally correct.

## Long-Horizon Structure

The task contains multiple logical phases rather than a fixed number
of actions.

Core phases:

1. Initial inspection
2. Incident localization
3. Dependency analysis
4. Root-cause hypothesis
5. Repair-strategy selection
6. Local intervention
7. Partial execution
8. Failure/result inspection
9. Downstream consequence analysis
10. Recovery or replanning
11. Full pipeline execution
12. Global validation
13. Final verification

## Branching

The initial incident can lead to different recovery families:

- Schema Failure
- Transformation Failure
- Data Quality Failure
- Resource / Execution Failure

Each recovery family contains its own long-horizon sequence.

## Nested Branching

Selected branches contain alternative recovery strategies.

Example:

Data Quality Failure
  -> filtering strategy
  -> reconstruction strategy

Transformation Failure
  -> patch transformation
  -> rollback and recompute

The agent must select a strategy based on observations.

## Stateful Consequences

Actions can modify persistent task state.

A repair can:

- fix one component,
- introduce a downstream inconsistency,
- invalidate an earlier assumption,
- require re-execution,
- require recovery,
- require replanning.

Therefore, the optimal trajectory cannot be reduced to a static action list.

## Hidden Evaluation

Hidden scenarios must not expose:

- hidden root causes,
- reference solutions,
- expected optimal trajectories,
- hidden repair choices,
- oracle answers.

Hidden scenario data belongs exclusively to the evaluator boundary.

## Success

Success requires global correctness.

Local component repair alone must not terminate the episode.

The final verification stage should require:

- required pipeline stages completed,
- final output valid,
- consistency checks passed,
- recovery obligations satisfied,
- no unresolved incident,
- final global verification passed.
