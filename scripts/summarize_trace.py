#!/usr/bin/env python3
"""Summarize a JSONL trace file: per-run and per-node aggregates.

Reads the trace written by `scripts/demo.py` (or any run that traced
to JSONL) and prints spans, tokens, cost, latency, and errors per run
and per worker node. Use `--json` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ops.summary import load_trace_records, summarize_trace


def _render(summary: dict) -> str:
    lines = []
    for run_id, run in summary["runs"].items():
        lines.append(f"run {run_id}: {run['spans']} spans, "
                     f"{run['tokens']} tokens, ${run['cost_usd']:.6f}, "
                     f"{run['latency_ms']} ms, {run['errors']} errors")
        for node, node_summary in run["nodes"].items():
            lines.append(
                f"  {node}: {node_summary['spans']} spans, "
                f"{node_summary['tokens']} tokens, "
                f"${node_summary['cost_usd']:.6f}, "
                f"{node_summary['latency_ms']} ms, "
                f"{node_summary['errors']} errors"
            )
    totals = summary["totals"]
    lines.append(
        f"total ({summary['run_count']} runs): {totals['spans']} spans, "
        f"{totals['tokens']} tokens, ${totals['cost_usd']:.6f}, "
        f"{totals['latency_ms']} ms, {totals['errors']} errors"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", default="traces/demo.jsonl")
    parser.add_argument("--run-id", default=None,
                        help="show only this run_id")
    parser.add_argument("--json", action="store_true",
                        help="print the summary as JSON")
    args = parser.parse_args()

    try:
        records, skipped = load_trace_records(args.trace_path)
    except FileNotFoundError:
        print(f"no trace file: {args.trace_path}", file=sys.stderr)
        return 1
    if args.run_id is not None:
        records = [r for r in records if str(r.get("run_id")) == args.run_id]
    summary = summarize_trace(records)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(_render(summary))
    if skipped:
        print(f"skipped {skipped} malformed/blank lines", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
