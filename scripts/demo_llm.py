#!/usr/bin/env python3
"""LLM demo: run a sample task through the graph with real LLM workers.

Requires an OpenAI-compatible API key in the environment; nothing is
sent anywhere without it. Set:

    LLM_API_KEY     (required)
    LLM_BASE_URL    (default: https://api.openai.com/v1)
    LLM_MODEL       (default: gpt-4o-mini)

The demo uses the researcher + summarizer presets under an LLM router,
with the same retry, budget, guardrail, and tracing ops as the offline
demo. Fails fast with a clear message when the key is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ops.graph import LLMRouter
from agent_ops.llm import LLMClient, researcher_worker, summarizer_worker
from agent_ops.policies import CostBudget, RetryPolicy
from agent_ops.runner import run_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="search and summarize: quarterly report")
    parser.add_argument("--trace-path", default="traces/demo_llm.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("LLM_API_KEY is not set; refusing to run without a key.")
        return 2
    client = LLMClient(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    workers = [researcher_worker(client), summarizer_worker(client)]
    outcome = run_task(
        args.task,
        LLMRouter([w.name for w in workers], client.complete),
        workers,
        retry=RetryPolicy(max_attempts=3),
        budget=CostBudget(max_tokens=10000, max_cost_usd=1.0),
        trace_path=args.trace_path,
    )
    if not outcome.ok:
        print(f"run failed: {outcome.error}")
        return 1
    for result in outcome.results:
        print(f"[{result['worker']}] {result['output']}")
    print(json.dumps({"trace_totals": outcome.trace_totals}, indent=2))
    print(f"trace: {args.trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
