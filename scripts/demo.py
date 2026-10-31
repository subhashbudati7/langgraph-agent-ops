#!/usr/bin/env python3
"""Offline demo: run a sample task through the supervised graph.

Uses deterministic fake workers only — no LLM calls, no API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ops.graph import KeywordRouter
from agent_ops.policies import CostBudget, RetryPolicy
from agent_ops.runner import run_task
from agent_ops.workers import FakeWorker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="search and summarize: quarterly report")
    parser.add_argument("--trace-path", default="traces/demo.jsonl")
    args = parser.parse_args()

    workers = [
        FakeWorker(name="researcher", output="research notes: revenue up 12%"),
        FakeWorker(name="summarizer", output="summary: revenue up 12% on strong demand"),
    ]
    outcome = run_task(
        args.task,
        KeywordRouter({"search": "researcher", "summarize": "summarizer"}),
        workers,
        retry=RetryPolicy(max_attempts=3),
        budget=CostBudget(max_tokens=1000, max_cost_usd=1.0),
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
