from __future__ import annotations

from dataclasses import asdict
from typing import Any

from branching_decision_recovery_v1.environment import BranchingDecisionRecoveryEnv
from branching_decision_recovery_v1.oracle import BranchingDecisionOracle
from branching_decision_recovery_v1.verifier import BranchingDecisionVerifier

from .hidden_scenarios import HiddenScenario, get_hidden_scenarios


def run_hidden_scenario(
    scenario: HiddenScenario,
) -> dict[str, Any]:
    env = BranchingDecisionRecoveryEnv()
    env.scenario_config = scenario.config
    oracle = BranchingDecisionOracle()
    verifier = BranchingDecisionVerifier()

    oracle_result = oracle.solve(env, seed=scenario.seed)
    verifier_result = verifier.verify(env.state)

    return {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "scenario_family": scenario.scenario_family,
            "topology": scenario.config.topology,
            "symptom_profile": scenario.config.symptom_profile,
            "consequence_delay": scenario.config.consequence_delay,
        },
        "oracle_solved": oracle_result.solved,
        "oracle_reason": oracle_result.reason,
        "verifier_verified": verifier_result.verified,
        "verifier_reason": verifier_result.reason,
        "terminal": env.state.terminal,
        "success": env.state.success,
        "logical_stage": env.state.logical_stage,
        "failure_detected": env.state.failure_detected,
        "recovery_required": env.state.recovery_required,
        "replanning_required": env.state.replanning_required,
        "validation_status": env.state.validation_status,
    }


def run_hidden_evaluation() -> dict[str, Any]:
    results = [
        run_hidden_scenario(scenario)
        for scenario in get_hidden_scenarios()
    ]

    oracle_pass = all(item["oracle_solved"] for item in results)
    verifier_pass = all(item["verifier_verified"] for item in results)

    return {
        "num_scenarios": len(results),
        "oracle_pass": oracle_pass,
        "verifier_pass": verifier_pass,
        "hidden_evaluation_pass": oracle_pass and verifier_pass,
        "results": results,
    }
