from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import time
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _json_safe(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))


@dataclass
class EpisodeStep:
    step: int
    action: str
    reward: float
    logical_stage: int
    terminal: bool
    success: bool
    observation: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class AgentEpisodeReport:
    episode_id: str
    task_id: str
    scenario_id: str | None
    seed: int | None

    status: str = "running"
    success: bool = False

    final_reward: float = 0.0
    total_reward: float = 0.0

    action_count: int = 0
    logical_stages_reached: int = 0
    recovery_count: int = 0
    failure_count: int = 0

    elapsed_seconds: float = 0.0

    token_usage: TokenUsage = field(default_factory=TokenUsage)

    reward_components: dict[str, float] = field(default_factory=dict)

    failures: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    recoveries: list[dict[str, Any]] = field(default_factory=list)

    trajectory: list[EpisodeStep] = field(default_factory=list)

    verifier_result: Any = None
    oracle_result: Any = None

    metrics: dict[str, float] = field(default_factory=dict)

    def add_step(
        self,
        action: str,
        reward: float,
        observation: Any,
        terminal: bool,
        success: bool,
        info: dict[str, Any] | None = None,
        elapsed_seconds: float = 0.0,
    ) -> None:
        info = dict(info or {})

        stage = observation.get("logical_stage", 0)

        step = EpisodeStep(
            step=len(self.trajectory) + 1,
            action=action,
            reward=float(reward),
            logical_stage=int(stage),
            terminal=bool(terminal),
            success=bool(success),
            observation=_json_safe(observation),
            info=_json_safe(info),
            elapsed_seconds=float(elapsed_seconds),
        )

        self.trajectory.append(step)

        self.total_reward += float(reward)
        self.final_reward = float(reward)
        self.action_count += 1
        self.logical_stages_reached = max(
            self.logical_stages_reached,
            int(stage),
        )

        if action == "recover":
            self.recovery_count += 1
            self.recoveries.append(
                {
                    "step": step.step,
                    "logical_stage": step.logical_stage,
                    "reward": step.reward,
                }
            )

        failure = info.get("failure")

        if failure not in (None, "", "none"):
            self.failure_count += 1
            self.failures.append(
                {
                    "step": step.step,
                    "type": failure,
                    "logical_stage": step.logical_stage,
                    "reward": step.reward,
                }
            )

        decision = info.get("decision")

        if decision not in (None, ""):
            self.decisions.append(
                {
                    "step": step.step,
                    "decision": decision,
                    "logical_stage": step.logical_stage,
                }
            )

    def add_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.token_usage.add(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def set_verifier_result(self, result: Any) -> None:
        self.verifier_result = _json_safe(result)

    def set_oracle_result(self, result: Any) -> None:
        self.oracle_result = _json_safe(result)

    def finalize(
        self,
        success: bool,
        status: str | None = None,
    ) -> None:
        self.success = bool(success)

        if status is None:
            status = "success" if success else "failure"

        self.status = status

        self.metrics = {
            "reward_per_step": (
                self.total_reward / self.action_count
                if self.action_count
                else 0.0
            ),
            "time_per_step": (
                self.elapsed_seconds / self.action_count
                if self.action_count
                else 0.0
            ),
            "tokens_per_step": (
                self.token_usage.total_tokens / self.action_count
                if self.action_count
                else 0.0
            ),
            "reward_per_token": (
                self.total_reward / self.token_usage.total_tokens
                if self.token_usage.total_tokens
                else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe(asdict(self))

        token_usage = dict(payload["token_usage"])
        token_usage["total_tokens"] = self.token_usage.total_tokens

        payload["token_usage"] = token_usage

        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            sort_keys=True,
        )

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            self.to_json(),
            encoding="utf-8",
        )
        return output


class EpisodeReporter:
    """
    Runtime reporter around an AgentForge environment.

    It does not modify environment semantics.
    It only records execution telemetry.
    """

    def __init__(
        self,
        task_id: str,
        scenario_id: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> None:
        self.report = AgentEpisodeReport(
            episode_id=episode_id or self._episode_id(task_id),
            task_id=task_id,
            scenario_id=scenario_id,
            seed=seed,
        )
        self._started_at: float | None = None

    @staticmethod
    def _episode_id(task_id: str) -> str:
        return f"{task_id}-{time.time_ns()}"

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self.report.status = "running"

    def record_step(
        self,
        action: str,
        reward: float,
        observation: Any,
        terminal: bool,
        success: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        if self._started_at is None:
            self.start()

        now = time.perf_counter()
        elapsed = now - self._started_at

        self.report.elapsed_seconds = elapsed

        self.report.add_step(
            action=action,
            reward=reward,
            observation=observation,
            terminal=terminal,
            success=success,
            info=info,
            elapsed_seconds=elapsed,
        )

    def record_environment_step(
        self,
        *,
        action: str,
        reward: float,
        logical_stage: int,
        terminal: bool,
        success: bool,
        observation=None,
        info=None,
        elapsed_seconds: float = 0.0,
    ):
        return self.record_step(
            action=action,
            reward=reward,
            observation=observation,
            info={
                **dict(info or {}),
                "logical_stage": logical_stage,
            },
            terminal=terminal,
            success=success,
            elapsed_seconds=elapsed_seconds,
        )


    def add_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.report.add_tokens(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def finalize(
        self,
        success: bool,
        status: str | None = None,
    ) -> AgentEpisodeReport:
        if self._started_at is not None:
            self.report.elapsed_seconds = (
                time.perf_counter() - self._started_at
            )

        self.report.finalize(
            success=success,
            status=status,
        )

        return self.report
