from agentforge.orchestration import OrchestrationContext


def test_context_creation_and_metadata_copy() -> None:
    context = OrchestrationContext(
        task_id="task-001",
        scenario_id="scenario-001",
        seed=42,
    )

    updated = context.with_metadata(mode="evaluation")

    assert context.task_id == "task-001"
    assert context.seed == 42
    assert context.metadata == {}
    assert updated.metadata == {"mode": "evaluation"}
