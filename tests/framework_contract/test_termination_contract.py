import pytest

from agentforge.contracts.termination import TerminationContract


def test_termination_and_truncation_are_exclusive() -> None:
    contract = TerminationContract()

    contract.validate(False, False)
    contract.validate(True, False)
    contract.validate(False, True)

    with pytest.raises(ValueError):
        contract.validate(True, True)


def test_flags_must_be_boolean() -> None:
    contract = TerminationContract()

    with pytest.raises(TypeError):
        contract.validate(1, False)

    with pytest.raises(TypeError):
        contract.validate(False, 1)
