from agentforge.contracts import (
    ActionContract,
    EnvironmentStateError,
    EvaluationError,
    FrameworkError,
    InfoContract,
    InvalidActionError,
    LifecycleContract,
    ObservationContract,
    RewardContract,
    ScenarioGenerationError,
    SeedContract,
    TaskConfigurationError,
    TaskContract,
    TerminationContract,
)


def test_contract_imports() -> None:
    contracts = [
        ActionContract,
        EnvironmentStateError,
        EvaluationError,
        FrameworkError,
        InfoContract,
        InvalidActionError,
        LifecycleContract,
        ObservationContract,
        RewardContract,
        ScenarioGenerationError,
        SeedContract,
        TaskConfigurationError,
        TaskContract,
        TerminationContract,
    ]

    assert len(contracts) == 14
