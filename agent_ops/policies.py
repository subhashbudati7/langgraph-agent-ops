"""Execution policies: retries with backoff, and per-run cost budgets."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agent_ops.workers import TransientWorkerError, Worker, WorkerResult


class BudgetExceeded(Exception):
    """Raised when a worker call would exceed the run's cost budget."""


@dataclass(frozen=True)
class RetryPolicy:
    """Retry transient worker failures with exponential backoff.

    Only `TransientWorkerError` is retried; other exceptions propagate
    immediately. `sleep` is injectable so tests never actually wait.
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    factor: float = 2.0
    max_delay_s: float = 60.0

    def delays(self) -> list[float]:
        """Delays between attempts (len == max_attempts - 1)."""
        return [
            min(self.base_delay_s * (self.factor**i), self.max_delay_s)
            for i in range(max(0, self.max_attempts - 1))
        ]


@dataclass
class CostBudget:
    """Per-run token and dollar caps, checked before each worker call."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0

    def check(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        if self.max_tokens is not None and self.tokens_used + tokens > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exceeded: {self.tokens_used + tokens} > {self.max_tokens}"
            )
        if self.max_cost_usd is not None and self.cost_usd + cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget exceeded: ${self.cost_usd + cost_usd:.4f} > ${self.max_cost_usd:.4f}"
            )

    def record(self, result: WorkerResult) -> None:
        self.tokens_used += result.tokens
        self.cost_usd += result.cost_usd


def run_worker_with_policy(
    worker: Worker,
    task: str,
    retry: RetryPolicy | None = None,
    budget: CostBudget | None = None,
    sleep=time.sleep,
) -> WorkerResult:
    """Run one worker under the retry policy and cost budget.

    The budget is checked after each successful attempt, once the real
    tokens/cost are known: a result that would breach the budget is
    discarded (not recorded) and `BudgetExceeded` is raised. A pre-call
    estimate check is a planned addition for LLM workers, whose cost can
    be estimated from prompt tokens before the call.
    """
    policy = retry or RetryPolicy()
    last_error: Exception | None = None
    for attempt, delay in enumerate([0.0, *policy.delays()]):
        if attempt > 0:
            sleep(delay)
        try:
            result = worker.run(task)
        except TransientWorkerError as exc:
            last_error = exc
            continue
        if budget is not None:
            budget.check(tokens=result.tokens, cost_usd=result.cost_usd)
            budget.record(result)
        return result
    assert last_error is not None  # max_attempts >= 1 by construction
    raise last_error
