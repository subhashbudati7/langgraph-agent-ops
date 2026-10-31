"""Worker protocol and deterministic stand-ins for real agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class WorkerError(Exception):
    """Base class for worker failures."""


class TransientWorkerError(WorkerError):
    """A failure worth retrying (rate limit, timeout, 5xx)."""


@dataclass(frozen=True)
class WorkerResult:
    worker: str
    output: str
    tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "worker": self.worker,
            "output": self.output,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
        }


class Worker(Protocol):
    """A worker agent: given a task, produce one result."""

    name: str

    def run(self, task: str) -> WorkerResult:
        ...


@dataclass
class FakeWorker:
    """Deterministic offline worker for tests and demos.

    Never calls an LLM. Optionally fails `fail_times` times with a
    transient error first, so retry behaviour can be tested end to end.
    """

    name: str
    output: str = ""
    tokens: int = 10
    cost_usd: float = 0.001
    fail_times: int = 0
    calls: int = field(default=0, init=False)

    def run(self, task: str) -> WorkerResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientWorkerError(f"{self.name} transient failure (call {self.calls})")
        return WorkerResult(
            worker=self.name,
            output=self.output or f"{self.name}: {task}",
            tokens=self.tokens,
            cost_usd=self.cost_usd,
        )
