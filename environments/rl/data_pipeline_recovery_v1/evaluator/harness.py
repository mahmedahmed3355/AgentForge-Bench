from __future__ import annotations

from dataclasses import dataclass

from data_pipeline_recovery_v1.actions import ActionKind
from data_pipeline_recovery_v1.decision_graph import (
    IncidentKind,
    RecoveryStrategy,
)
from data_pipeline_recovery_v1.environment import DataPipelineRecoveryEnv
from data_pipeline_recovery_v1.oracle import PipelineRecoveryOracle
from data_pipeline_recovery_v1.verifier import PipelineRecoveryVerifier

from .hidden_scenarios import HiddenScenario, load_hidden_scenarios


@dataclass(frozen=True)
class EvaluationResult:
    scenario_id: str
    seed: int
    incident_family: str
    oracle_success: bool
    verifier_success: bool
    terminal: bool
    logical_stage: int
    action_count: int


def _step(
    env: DataPipelineRecoveryEnv,
    action: ActionKind,
    **kwargs,
):
    if env.state.terminal:
        raise RuntimeError(
            f"Reference trajectory attempted action {action.value!r} "
            "after terminal state."
        )
    result = env.step(action, **kwargs)
    if result.info.get("verification_failed"):
        raise RuntimeError(
            f"Reference trajectory triggered premature final verification: "
            f"{result.info}"
        )
    return result


def _setup(
    env: DataPipelineRecoveryEnv,
    branch: str,
    strategy: str,
) -> None:
    _step(env, ActionKind.INSPECT_PIPELINE)
    _step(env, ActionKind.ANALYZE_FAILURE)
    _step(env, ActionKind.SELECT_BRANCH, branch=branch)
    _step(env, ActionKind.SELECT_STRATEGY, strategy=strategy)


def _reference_schema(env: DataPipelineRecoveryEnv) -> None:
    _setup(
        env,
        "schema_recovery",
        RecoveryStrategy.REPAIR_UPSTREAM.value,
    )

    _step(
        env,
        ActionKind.MODIFY_COMPONENT,
        component="schema_validation",
    )

    _step(env, ActionKind.RUN_PIPELINE)
    _step(
        env,
        ActionKind.INSPECT_COMPONENT,
        component="cleaning",
    )
    _step(
        env,
        ActionKind.INSPECT_COMPONENT,
        component="transformation",
    )
    _step(
        env,
        ActionKind.INSPECT_COMPONENT,
        component="aggregation",
    )
    _step(env, ActionKind.CHECK_SCHEMA)
    _step(env, ActionKind.CHECK_OUTPUT)
    _step(env, ActionKind.FINAL_VERIFY)

def _reference_transformation(env: DataPipelineRecoveryEnv) -> None:
    _setup(
        env,
        "transformation_recovery",
        RecoveryStrategy.PATCH_TRANSFORM.value,
    )

    _step(
        env,
        ActionKind.MODIFY_COMPONENT,
        component="transformation",
    )

    assert env.state.replanning_required, (
        "patch_transform reference must require replanning"
    )

    _step(env, ActionKind.MODIFY_CONFIG)
    _step(env, ActionKind.REPLAN)

    _step(env, ActionKind.RUN_PIPELINE)

    if env.state.recovery_required:
        _step(env, ActionKind.RECOVER)

    _step(env, ActionKind.RUN_PIPELINE)

    if env.state.recovery_required:
        _step(env, ActionKind.RECOVER)

    _step(env, ActionKind.RUN_PIPELINE)
    _step(env, ActionKind.FINAL_VERIFY)


def _reference_data_quality(env: DataPipelineRecoveryEnv) -> None:
    _setup(
        env,
        "data_quality_recovery",
        RecoveryStrategy.FILTER_INVALID.value,
    )

    _step(
        env,
        ActionKind.MODIFY_COMPONENT,
        component="cleaning",
    )

    assert env.state.replanning_required, (
        "filter_invalid reference must require replanning"
    )

    _step(env, ActionKind.MODIFY_CONFIG)
    _step(env, ActionKind.REPLAN)

    _step(env, ActionKind.RUN_PIPELINE)

    if env.state.recovery_required:
        _step(env, ActionKind.RECOVER)

    _step(env, ActionKind.RUN_PIPELINE)

    if not env.state.output_valid:
        _step(env, ActionKind.CHECK_OUTPUT)

    _step(env, ActionKind.FINAL_VERIFY)


def _reference_resource(env: DataPipelineRecoveryEnv) -> None:
    _setup(
        env,
        "resource_recovery",
        RecoveryStrategy.OPTIMIZE_EXECUTION.value,
    )

    _step(
        env,
        ActionKind.MODIFY_COMPONENT,
        component="transformation",
    )
    _step(env, ActionKind.MODIFY_CONFIG)
    _step(env, ActionKind.REPLAN)

    _step(env, ActionKind.RUN_PIPELINE)
    _step(env, ActionKind.RECOVER)
    _step(env, ActionKind.RUN_PIPELINE)

    _step(env, ActionKind.CHECK_SCHEMA)
    _step(env, ActionKind.CHECK_OUTPUT)
    _step(env, ActionKind.FINAL_VERIFY)


_REFERENCE_SOLVERS = {
    IncidentKind.SCHEMA: _reference_schema,
    IncidentKind.TRANSFORMATION: _reference_transformation,
    IncidentKind.DATA_QUALITY: _reference_data_quality,
    IncidentKind.RESOURCE: _reference_resource,
}


def _solve_reference(env: DataPipelineRecoveryEnv) -> None:
    solver = _REFERENCE_SOLVERS[env.state.incident]
    solver(env)


def evaluate_hidden_scenario(scenario: HiddenScenario) -> EvaluationResult:
    env = DataPipelineRecoveryEnv(
        episode_id=scenario.scenario_id,
        incident=scenario.incident_family,
    )
    env.reset()

    _solve_reference(env)

    oracle = PipelineRecoveryOracle()

    oracle_result = oracle.evaluate(
        scenario.incident_family,
        branch=env.state.selected_branch,
        strategy=env.state.selected_strategy,
        logical_stage=env.state.logical_stage,
        all_components_valid=env.state.all_components_valid,
        output_valid=env.state.output_valid,
        recovery_required=env.state.recovery_required,
        replanning_required=env.state.replanning_required,
        downstream_inconsistency=env.state.downstream_inconsistency,
        pipeline_run_completed=env.state.pipeline_run_completed,
    )

    verifier_result = PipelineRecoveryVerifier().verify(env.state)

    return EvaluationResult(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        incident_family=scenario.incident_family.value,
        oracle_success=oracle_result.success,
        verifier_success=verifier_result.success,
        terminal=env.state.terminal,
        logical_stage=env.state.logical_stage,
        action_count=env.state.action_count,
    )


def run_hidden_evaluation() -> tuple[EvaluationResult, ...]:
    return tuple(
        evaluate_hidden_scenario(scenario)
        for scenario in load_hidden_scenarios()
    )
