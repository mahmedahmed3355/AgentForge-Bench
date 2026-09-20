import pytest

from agentforge.contracts.info import InfoContract


def test_safe_info_is_accepted() -> None:
    InfoContract().validate(
        {
            "episode_id": "ep-1",
            "step": 2,
            "phase": "analysis",
        }
    )


@pytest.mark.parametrize(
    "key",
    [
        "oracle_action",
        "oracle_trajectory",
        "hidden_solution",
        "ground_truth",
        "reference_trajectory",
        "verifier_answer",
        "correct_action",
    ],
)
def test_evaluator_truth_is_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        InfoContract().validate({key: "secret"})
