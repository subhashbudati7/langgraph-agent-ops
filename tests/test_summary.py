"""Tests for trace summarization (agent_ops/summary.py)."""

from __future__ import annotations

import json

import pytest

from agent_ops.summary import load_trace_records, summarize_trace


def _record(run_id="r1", node="researcher", **overrides):
    record = {
        "run_id": run_id,
        "node": node,
        "status": "ok",
        "tokens": 10,
        "cost_usd": 0.001,
        "latency_ms": 2.5,
    }
    record.update(overrides)
    return record


def test_summarize_single_run_totals():
    summary = summarize_trace([_record(), _record(node="summarizer", tokens=20)])
    assert summary["run_count"] == 1
    assert summary["totals"] == {
        "spans": 2,
        "tokens": 30,
        "cost_usd": 0.002,
        "latency_ms": 5.0,
        "errors": 0,
    }


def test_summarize_groups_by_run_and_node():
    records = [
        _record(run_id="r1", node="researcher"),
        _record(run_id="r1", node="summarizer"),
        _record(run_id="r2", node="researcher", status="error", error="boom"),
    ]
    summary = summarize_trace(records)
    assert summary["run_count"] == 2
    assert set(summary["runs"]) == {"r1", "r2"}
    assert summary["runs"]["r1"]["nodes"]["researcher"]["spans"] == 1
    assert summary["runs"]["r2"]["errors"] == 1
    assert summary["totals"]["errors"] == 1


def test_summarize_missing_fields_default_to_zero():
    summary = summarize_trace([{"run_id": "r1", "node": "researcher"}])
    assert summary["totals"]["tokens"] == 0
    assert summary["totals"]["cost_usd"] == 0.0
    assert summary["totals"]["latency_ms"] == 0.0


def test_summarize_records_without_run_id_grouped_as_unknown():
    summary = summarize_trace([{"node": "researcher", "status": "ok"}])
    assert summary["runs"]["unknown"]["spans"] == 1


def test_load_trace_records_skips_bad_lines(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(_record()) + "\n"
        + "not json\n"
        + "\n"
        + "[1, 2]\n"
        + json.dumps(_record(node="summarizer")) + "\n",
        encoding="utf-8",
    )
    records, skipped = load_trace_records(path)
    assert len(records) == 2
    assert skipped == 3
    assert records[1]["node"] == "summarizer"


def test_load_trace_records_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trace_records(tmp_path / "nope.jsonl")
