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
- **LLM workers** (`agent_ops/llm.py`): `LLMClient` (stdlib-only,
  OpenAI-compatible chat completions with usage/cost accounting and
  429/5xx mapped to transient errors), `LLMWorker`, and
  researcher/summarizer/critic/planner/editor presets. `LLMRouter`
  (`graph.py`) asks an LLM for the next worker and ends the run on
  any unparseable reply; `RetryingRouter` retries transient router
  failures with backoff and falls back to a fallback router (default:
  done) when attempts run out. `CostBudget.precheck` fails fast on
  an over-budget estimate before a call ever bills.
- **Policies** (`agent_ops/policies.py`): `RetryPolicy` (max attempts,
  exponential backoff) and `CostBudget` (token and dollar caps) enforced
  per run, with `BudgetExceeded` raised before a call that would blow
  the budget.
- **Guardrails** (`agent_ops/guardrails.py`): input/output checks —
  blocked-pattern scan, output length cap — raising
  `GuardrailViolation` instead of letting bad content through.
  Outputs are checked inside the graph node, so a violating result
  (SSN-shaped number, OpenAI/AWS-style key) never enters state or
  the trace as a success; `run_task()` reports the violation on the
  outcome. Input violations still propagate to the caller.
- **HTTP API** (`agent_ops/api.py`): `POST /run` runs a task through
  the supervisor graph and `GET /health` is the liveness probe.
  Offline-first — the default stack needs no key — and failures
  (guardrail, budget) are reported as a 200 with an `error` field,
  never a 500. `scripts/serve_api.py` serves the API, wiring real
  LLM workers when `LLM_API_KEY` is set.
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
| Trace summarizer | Per-run/node totals from JSONL (`scripts/summarize_trace.py`) | Done |
| Run report | JSON report per run (`scripts/run_report.py`) | Planned |
| Real LLM workers | OpenAI-compatible client | Done |
| LLM routing | `LLMRouter` with fail-to-done | Done |
| Router retry | `RetryingRouter`, backoff + fallback | Done |
| Pre-call budget estimate | `CostBudget.precheck` | Done |
| Output guardrails in-graph | Checked per worker node, key scan | Done |
| HTTP API | `POST /run`, `GET /health` (offline stack) | Done |
| Live demo | Hosted API | Planned (Nov 27 deploy) |

## Run it

```bash
pip install -e ".[dev]"
python scripts/demo.py --task "summarize: quarterly report"
python scripts/summarize_trace.py --trace-path traces/demo.jsonl
python -m pytest
ruff check .
```

Summarize any JSONL trace after a run — per run and per worker node,
with `--json` for machine-readable output and `--run-id` to filter to
one run:

Serve the same graph over HTTP (offline stack, no key needed):

```bash
pip install -e ".[api]"
python scripts/serve_api.py  # POST localhost:8000/run
curl -X POST localhost:8000/run -d '{"task": "summarize: quarterly report"}'
```

With an OpenAI-compatible key, the same ops layer runs real workers:

```bash
LLM_API_KEY=... LLM_MODEL=gpt-4o-mini python scripts/demo_llm.py
```

`demo_llm.py` refuses to run without `LLM_API_KEY` set, and never logs
the key. `serve_api.py` likewise serves the offline stack when the
key is unset, and real LLM workers (planner, researcher, summarizer,
editor) when it is.

## Status

Week 1 (Oct 31, 2026): scaffold, supervisor graph, policies,
guardrails, tracing, and offline demo runner. All tests pass on the
offline stack; no LLM calls are made anywhere in this release.

Week 2 (Nov 7, 2026): router retry with fallback, output guardrails
enforced per worker node (PII + leaked-key scan), planner/editor
worker presets, and a FastAPI `/run` endpoint with an offline-first
serve script. 45 tests passing, ruff clean; the API was verified
live against the offline stack.
