from agentforge.orchestration import EpisodeOrchestrator


def test_episode_orchestrator_runs_to_terminal_state() -> None:
    state = {"value": 0}

    def reset():
        state["value"] = 0
        return state["value"]

    def step(observation):
        state["value"] += 1
        terminated = state["value"] >= 3
        return (
            state["value"],
            1.0,
            terminated,
            False,
            {"value": state["value"]},
        )

    runner = EpisodeOrchestrator(
        reset=reset,
        step=step,
        is_success=lambda observation: observation == 3,
    )

    result = runner.run(
        task_id="task-001",
        scenario_id="scenario-001",
    )

    assert result.steps == 3
    assert result.total_reward == 3.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.success is True
