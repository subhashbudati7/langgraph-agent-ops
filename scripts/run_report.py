#!/usr/bin/env python3
"""Run a task through the offline stack and write a JSON run report.

Runs the same deterministic demo stack as `scripts/demo.py`, then
writes a `RunReport` (outcome + per-node trace summary) to
`reports/<run_id>.json` and prints a one-line summary. Reports are
git-ignored — they are run artifacts, not source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ops.graph import KeywordRouter
from agent_ops.policies import CostBudget, RetryPolicy
from agent_ops.report import RunReport
from agent_ops.runner import run_task
from agent_ops.summary import load_trace_records
from agent_ops.workers import FakeWorker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="search and summarize: quarterly report")
    parser.add_argument("--trace-path", default="traces/demo.jsonl")
    parser.add_argument("--report-dir", default="reports")
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
    records, _ = load_trace_records(args.trace_path)
    report = RunReport.from_outcome(outcome, records)
    report_path = report.write_json(Path(args.report_dir) / f"{outcome.run_id}.json")
    totals = report.trace or {}
    print(
        f"run {outcome.run_id}: ok, {len(outcome.results)} workers, "
        f"{totals.get('tokens', 0)} tokens, "
        f"${totals.get('cost_usd', 0.0):.6f}, "
        f"{totals.get('latency_ms', 0.0)} ms"
    )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
