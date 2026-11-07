#!/usr/bin/env python3
"""Serve the supervisor graph over HTTP (uvicorn).

Without LLM_API_KEY, serves the offline stack: keyword router plus
fake workers, so the endpoint works with no key and no network. With
LLM_API_KEY set, serves the real LLM workers (researcher, summarizer,
planner, editor) under an LLM router wrapped in RetryingRouter, with
the same retry, budget, guardrail, and tracing ops as the demos.

Environment:

    LLM_API_KEY     (optional — offline stack when unset)
    LLM_BASE_URL    (default: https://api.openai.com/v1)
    LLM_MODEL       (default: gpt-4o-mini)
    API_HOST        (default: 127.0.0.1)
    API_PORT        (default: 8000)

Run:  python scripts/serve_api.py
Then: curl -X POST localhost:8000/run -d '{"task": "summarize: quarterly report"}'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from agent_ops.api import create_app
from agent_ops.graph import LLMRouter, RetryingRouter
from agent_ops.llm import (
    LLMClient,
    editor_worker,
    planner_worker,
    researcher_worker,
    summarizer_worker,
)


def build_app():
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("LLM_API_KEY not set — serving the offline demo stack.")
        return create_app()
    client = LLMClient(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    workers = [
        planner_worker(client),
        researcher_worker(client),
        summarizer_worker(client),
        editor_worker(client),
    ]
    router = RetryingRouter(
        LLMRouter([w.name for w in workers], client.complete),
        max_attempts=3,
    )
    print(f"serving LLM workers {[w.name for w in workers]} with model {client.model}.")
    return create_app(router=router, workers=workers)


def main() -> int:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
