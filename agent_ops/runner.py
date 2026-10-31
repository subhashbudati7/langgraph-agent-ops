"""Run a task end to end: guardrails -> graph -> output guardrails."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from agent_ops.graph import Router, build_graph
from agent_ops.guardrails import check_input, check_output
from agent_ops.policies import BudgetExceeded, CostBudget, RetryPolicy
from agent_ops.tracing import TraceRecorder
from agent_ops.workers import Worker, WorkerError


@dataclass
class RunOutcome:
    task: str
    results: list[dict] = field(default_factory=list)
    error: str | None = None
    trace_totals: dict | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_task(
    task: str,
    router: Router,
    workers: list[Worker],
    retry: RetryPolicy | None = None,
    budget: CostBudget | None = None,
    trace_path: str | None = None,
) -> RunOutcome:
    """Run one task through the supervised graph.

    Input guardrail violations propagate (the caller supplied a bad
    task); worker failures and budget breaches are reported on the
    outcome instead, so the caller can log and continue.
    """
    check_input(task)
    recorder = TraceRecorder(trace_path, run_id=str(uuid.uuid4())) if trace_path else None
    app = build_graph(router, workers, retry=retry, budget=budget, recorder=recorder)
    try:
        final = app.invoke({"task": task, "route": "", "results": [], "error": None})
    except (WorkerError, BudgetExceeded) as exc:
        return RunOutcome(
            task=task, error=str(exc),
            trace_totals=recorder.totals() if recorder else None,
        )
    results = final.get("results", [])
    for result in results:
        check_output(result["output"])
    return RunOutcome(
        task=task, results=results,
        trace_totals=recorder.totals() if recorder else None,
    )
