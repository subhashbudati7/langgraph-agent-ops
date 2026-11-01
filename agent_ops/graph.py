"""Supervisor graph: route a task to workers until done."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from langgraph.graph import END, START, StateGraph

from agent_ops.policies import CostBudget, RetryPolicy, run_worker_with_policy
from agent_ops.state import AgentState
from agent_ops.tracing import TraceRecorder
from agent_ops.workers import Worker


class Router(Protocol):
    """Decides the next worker, or "done", from the task and results so far."""

    def route(self, task: str, results: list[dict]) -> str:
        ...


class KeywordRouter:
    """Deterministic offline router.

    Maps keywords in the task to worker names and routes to each matching
    worker at most once, in mapping order. When every matching worker has
    run (or nothing matches), returns "done". Tasks with no matching
    keyword fall back to `default` if one is set.
    """

    def __init__(self, mapping: dict[str, str], default: str | None = None) -> None:
        self.mapping = mapping
        self.default = default

    def route(self, task: str, results: list[dict]) -> str:
        done_workers = {r["worker"] for r in results}
        lowered = task.lower()
        for keyword, worker in self.mapping.items():
            if keyword in lowered and worker not in done_workers:
                return worker
        if self.default is not None and self.default not in done_workers:
            return self.default
        return "done"


class LLMRouter:
    """Supervisor router that asks an LLM for the next worker.

    `complete` receives one user message listing the available workers,
    the task, and the workers already run, and must reply with a worker
    name or "done". Any unparseable reply, a repeat of an already-run
    worker, or an unknown name routes to "done" — a confused router
    ends the run instead of looping forever or dispatching nowhere.
    """

    def __init__(self, worker_names: list[str], complete: Callable[[list[dict]], str]):
        self.worker_names = worker_names
        self.complete = complete

    def route(self, task: str, results: list[dict]) -> str:
        done_workers = {r["worker"] for r in results}
        available = [w for w in self.worker_names if w not in done_workers]
        prompt = (
            f"Task: {task}\n"
            f"Available workers: {', '.join(available) or 'none'}\n"
            f"Already run: {', '.join(sorted(done_workers)) or 'none'}\n"
            "Reply with exactly one available worker name, or 'done' if the task is complete."
        )
        choice = self.complete([{"role": "user", "content": prompt}]).strip()
        if choice in available:
            return choice
        return "done"


def build_graph(
    router: Router,
    workers: list[Worker],
    retry: RetryPolicy | None = None,
    budget: CostBudget | None = None,
    recorder: TraceRecorder | None = None,
    estimate_tokens: Callable[[str], int] | None = None,
):
    """Build the supervisor graph for the given router and workers.

    Flow: START -> supervisor -> (worker | END); each worker returns to
    the supervisor, which routes again or finishes. Worker calls run
    under the retry policy and cost budget, and are traced per node when
    a recorder is given.
    """
    by_name = {w.name: w for w in workers}
    if len(by_name) != len(workers):
        raise ValueError("worker names must be unique")

    graph = StateGraph(AgentState)

    def supervisor(state: AgentState) -> dict:
        return {"route": router.route(state["task"], state.get("results", []))}

    graph.add_node("supervisor", supervisor)

    for worker in workers:
        def make_node(w: Worker):
            def node(state: AgentState) -> dict:
                if recorder is not None:
                    with recorder.span(w.name) as span_record:
                        result = run_worker_with_policy(
                            w, state["task"], retry, budget,
                            estimate_tokens=estimate_tokens,
                        )
                        span_record["tokens"] = result.tokens
                        span_record["cost_usd"] = result.cost_usd
                else:
                    result = run_worker_with_policy(
                        w, state["task"], retry, budget,
                        estimate_tokens=estimate_tokens,
                    )
                return {"results": [result.to_dict()]}

            return node

        graph.add_node(worker.name, make_node(worker))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {w.name: w.name for w in workers} | {"done": END},
    )
    for worker in workers:
        graph.add_edge(worker.name, "supervisor")
    return graph.compile()
