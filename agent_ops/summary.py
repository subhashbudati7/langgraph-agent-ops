"""Summarize JSONL traces: per-run and per-node aggregates.

Pure functions over trace records, so the same summary powers the
CLI (`scripts/summarize_trace.py`) and the run report
(`agent_ops/report.py`). Records are plain dicts as written by
`TraceRecorder.span` — `run_id`, `node`, `status`, `tokens`,
`cost_usd`, `latency_ms`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_trace_records(path: str | Path) -> tuple[list[dict], int]:
    """Load trace records from a JSONL file.

    Returns ``(records, skipped)`` where ``skipped`` counts blank
    lines and lines that are not valid JSON objects — a partially
    corrupted trace still yields a summary instead of crashing.
    """
    records: list[dict] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            records.append(record)
    return records, skipped


def summarize_trace(records: list[dict]) -> dict:
    """Aggregate trace records into per-run summaries plus totals.

    Runs are grouped by ``run_id`` (records without one land in
    ``"unknown"``). Each run summary carries span count, per-node
    breakdowns, and totals; the top level carries the same totals
    across all runs.
    """
    by_run: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_run[str(record.get("run_id") or "unknown")].append(record)

    runs = {run_id: _summarize_run(run_records) for run_id, run_records in by_run.items()}
    totals = _aggregate(records)
    return {"runs": runs, "totals": totals, "run_count": len(runs)}


def _summarize_run(records: list[dict]) -> dict:
    """One run's summary: totals plus a per-node breakdown."""
    by_node: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_node[str(record.get("node") or "unknown")].append(record)
    nodes = {node: _aggregate(node_records) for node, node_records in by_node.items()}
    summary = _aggregate(records)
    summary["nodes"] = nodes
    return summary


def _aggregate(records: list[dict]) -> dict:
    """Totals across a set of records: spans, tokens, cost, latency, errors."""
    return {
        "spans": len(records),
        "tokens": sum(int(r.get("tokens") or 0) for r in records),
        "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in records), 6),
        "latency_ms": round(sum(float(r.get("latency_ms") or 0.0) for r in records), 3),
        "errors": sum(1 for r in records if r.get("status") == "error"),
    }
