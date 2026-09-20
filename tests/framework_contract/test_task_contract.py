import pytest

from agentforge.contracts.task import TaskContract


def test_task_contract_accepts_valid_identity() -> None:
    contract = TaskContract(
        task_id="demo-task",
        version="1.0.0",
        max_episode_steps=100,
    )

    assert contract.task_id == "demo-task"
    assert contract.version == "1.0.0"
    assert contract.max_episode_steps == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_id": "", "version": "1.0.0", "max_episode_steps": 10},
        {"task_id": "demo", "version": "", "max_episode_steps": 10},
        {"task_id": "demo", "version": "1.0.0", "max_episode_steps": 0},
        {"task_id": "demo", "version": "1.0.0", "max_episode_steps": -1},
    ],
)
def test_task_contract_rejects_invalid_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TaskContract(**kwargs)
