# langgraph-agent-ops

Multi-agent workflow with production ops: a supervisor routes tasks to
worker agents, and every worker call runs under retry/backoff, a cost
budget, input/output guardrails, and a JSONL trace.

## Problem

Multi-agent demos usually show the happy path: one supervisor, a few
workers, no failures. Production runs need the opposite — workers fail
transiently, costs run away, bad inputs slip through, and nobody can
reconstruct what happened after the fact. This project is the ops layer
that makes a supervisor/worker graph safe to run unattended.

## Architecture

```
task ──► guardrails (input) ──► supervisor ──► worker ──┐
                                        │               │
                                        └──── loop ─────┘
                                            │
                                   guardrails (output)
                                            │
                                   trace (JSONL) + budget
```

- **Supervisor** (`agent_ops/graph.py`): a LangGraph `StateGraph` that
  routes each task to a worker and decides, from the worker's result,
  whether to route again or finish. Routing is a pluggable `Router`
  protocol; the built-in `KeywordRouter` is deterministic and offline.
- **Workers** (`agent_ops/workers.py`): a `Worker` protocol returning a
  `WorkerResult` (output, tokens used, cost). `FakeWorker` stands in for
  real agents in tests and demos; it never calls an LLM.
- **Policies** (`agent_ops/policies.py`): `RetryPolicy` (max attempts,
  exponential backoff) and `CostBudget` (token and dollar caps) enforced
  per run, with `BudgetExceeded` raised before a call that would blow
  the budget.
- **Guardrails** (`agent_ops/guardrails.py`): input/output checks —
  blocked-pattern scan, output length cap — raising
  `GuardrailViolation` instead of letting bad content through.
- **Tracing** (`agent_ops/tracing.py`): one JSON record per node
  invocation (node, status, tokens, cost, latency) appended to a JSONL
  file, so any run can be replayed from its trace.
- **Runner** (`agent_ops/runner.py`): `run_task()` wires the pieces in
  order: input guardrails → graph → output guardrails → trace.

## Ops table (week 1, offline stack)

| Concern | Mechanism | Status |
| --- | --- | --- |
| Routing | Supervisor + `KeywordRouter` | Done |
| Retries | `RetryPolicy`, exponential backoff | Done |
| Cost caps | `CostBudget` per run | Done |
| Guardrails | Input/output checks | Done |
| Tracing | JSONL spans per node | Done |
| Real LLM workers | OpenAI-compatible client | Planned |
| Live demo | Hosted API | Planned (Nov 27 deploy) |

## Run it

```bash
pip install -e ".[dev]"
python scripts/demo.py --task "summarize: quarterly report"
python -m pytest
ruff check .
```

## Status

Week 1 (Oct 31, 2026): scaffold, supervisor graph, policies,
guardrails, tracing, and offline demo runner. All tests pass on the
offline stack; no LLM calls are made anywhere in this release.
