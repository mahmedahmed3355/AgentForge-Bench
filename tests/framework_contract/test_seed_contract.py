import pytest

from agentforge.contracts.seed import SeedContract


def test_seed_accepts_int_or_none() -> None:
    contract = SeedContract()

    contract.validate(None)
    contract.validate(0)
    contract.validate(1234)


def test_seed_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        SeedContract().validate("1234")
