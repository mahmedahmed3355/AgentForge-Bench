from agentforge.orchestration import (
    EpisodeOrchestrator,
    OrchestrationContext,
    Orchestrator,
)


def test_top_level_orchestrator_runs_multiple_episodes() -> None:
    def factory():
        state = {"value": 0}

        def reset():
            state["value"] = 0
            return 0

        def step(observation):
            state["value"] += 1
            terminated = state["value"] >= 2
            return state["value"], 1.0, terminated, False, {}

        return EpisodeOrchestrator(
            reset=reset,
            step=step,
            is_success=lambda observation: observation == 2,
            max_steps=10,
        )

    orchestrator = Orchestrator(factory)

    result = orchestrator.evaluate(
        context=OrchestrationContext(
            task_id="task-001",
            scenario_id="scenario-001",
            seed=123,
        ),
        episodes=3,
    )

    assert len(result.episodes) == 3
    assert result.summary.episodes == 3
    assert result.summary.successful_episodes == 3
    assert result.summary.success_rate == 1.0
    assert result.summary.mean_reward == 2.0
