"""HTTP API for the supervisor graph: POST /run, GET /health.

Offline-first: the default stack is a keyword router plus fake
workers, so the endpoint works with no API key and no network — the
same deterministic demo as `scripts/demo.py`, over HTTP. To serve
real LLM workers, pass them explicitly to `create_app` (see
`scripts/serve_api.py`).

Requires the `api` extra: `pip install langgraph-agent-ops[api]`.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent_ops.graph import KeywordRouter, build_graph
from agent_ops.policies import CostBudget, RetryPolicy
from agent_ops.runner import run_task
from agent_ops.workers import FakeWorker, Worker


class RunRequest(BaseModel):
    """One task to run through the supervisor graph."""

    task: str = Field(min_length=1, max_length=4000)


class RunResponse(BaseModel):
    """Outcome of a run: results, or an error when the run failed."""

    task: str
    results: list[dict] = []
    error: str | None = None
    trace_totals: dict | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def default_workers() -> list[Worker]:
    """Offline demo stack: deterministic workers, no LLM calls."""
    return [FakeWorker(name="researcher"), FakeWorker(name="summarizer")]


def default_router() -> KeywordRouter:
    return KeywordRouter(
        {"search": "researcher", "research": "researcher", "summarize": "summarizer"},
        default="summarizer",
    )


def create_app(
    router: KeywordRouter | None = None,
    workers: list[Worker] | None = None,
    retry: RetryPolicy | None = None,
    budget: CostBudget | None = None,
) -> FastAPI:
    """Build the API app.

    LLM-backed serving wires LLM workers explicitly via `workers`;
    the default stack stays offline and deterministic.
    """
    _router = router or default_router()
    _workers = workers if workers is not None else default_workers()
    _retry = retry or RetryPolicy(max_attempts=2)
    _budget = budget or CostBudget(max_tokens=10_000, max_cost_usd=1.0)
    # Fail fast at startup on a bad worker set, not on the first request.
    build_graph(_router, _workers, retry=_retry, budget=_budget)

    app = FastAPI(title="langgraph-agent-ops", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/run", response_model=RunResponse)
    def run(request: RunRequest) -> RunResponse:
        outcome = run_task(
            request.task, _router, _workers, retry=_retry, budget=_budget
        )
        return RunResponse(
            task=outcome.task,
            results=outcome.results,
            error=outcome.error,
            trace_totals=outcome.trace_totals,
        )

    return app


app = create_app()
