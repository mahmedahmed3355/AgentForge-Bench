from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .models import (
    Action,
    Observation,
    OracleResult,
    RewardResult,
    Scenario,
    StepResult,
    TaskData,
    VerificationResult,
)


StateT = TypeVar("StateT")


class Task(ABC, Generic[StateT]):
    def __init__(self, task_data: TaskData) -> None:
        self.task_data = task_data

    @abstractmethod
    def reset(self, scenario: Scenario) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Observation:
        raise NotImplementedError


class Taskset(ABC):
    @abstractmethod
    def load(self) -> list[TaskData]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.load())


class Verifier(ABC):
    @abstractmethod
    def verify(
        self,
        task: Task,
        scenario: Scenario,
    ) -> VerificationResult:
        raise NotImplementedError


class Oracle(ABC):
    @abstractmethod
    def solve(
        self,
        task: Task,
        scenario: Scenario,
    ) -> OracleResult:
        raise NotImplementedError


class RewardFunction(ABC):
    @abstractmethod
    def compute(
        self,
        observation: Observation,
        action: Action,
        result: StepResult,
    ) -> RewardResult:
        raise NotImplementedError


class ScenarioGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        split: str,
        seed: int,
    ) -> Scenario:
        raise NotImplementedError
