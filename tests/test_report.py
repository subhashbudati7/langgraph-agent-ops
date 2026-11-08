"""Tests for run reports (agent_ops/report.py)."""

from __future__ import annotations

import json

from agent_ops.graph import KeywordRouter
from agent_ops.policies import CostBudget, RetryPolicy
from agent_ops.report import RunReport
from agent_ops.runner import RunOutcome, run_task
from agent_ops.summary import load_trace_records
from agent_ops.workers import FakeWorker


def _outcome(**overrides):
    outcome = RunOutcome(
        task="search and summarize: quarterly report",
        results=[
            {"worker": "researcher", "output": "notes"},
            {"worker": "summarizer", "output": "summary"},
        ],
        trace_totals={"spans": 2, "tokens": 20, "cost_usd": 0.002,
                      "latency_ms": 0.059, "errors": 0},
        run_id="r1",
    )
    for key, value in overrides.items():
        setattr(outcome, key, value)
    return outcome


def _records():
    return [
        {"run_id": "r1", "node": "researcher", "status": "ok",
         "tokens": 10, "cost_usd": 0.001, "latency_ms": 0.032},
        {"run_id": "r1", "node": "summarizer", "status": "ok",
         "tokens": 10, "cost_usd": 0.001, "latency_ms": 0.027},
        {"run_id": "other", "node": "researcher", "status": "ok",
         "tokens": 99, "cost_usd": 0.099, "latency_ms": 9.9},
    ]


def test_report_from_successful_outcome_with_records():
    report = RunReport.from_outcome(_outcome(), _records())
    assert report.ok is True
    assert report.error is None
    assert report.workers == ["researcher", "summarizer"]
    assert report.trace is not None
    assert report.trace["spans"] == 2
    assert report.trace["tokens"] == 20
    assert report.trace["nodes"]["researcher"]["spans"] == 1
    assert report.generated_at


def test_report_ignores_other_runs_records():
    report = RunReport.from_outcome(_outcome(), _records())
    assert report.trace is not None
    assert report.trace["tokens"] == 20  # not 119


def test_report_from_failed_outcome():
    report = RunReport.from_outcome(_outcome(error="budget exceeded", results=[]))
    assert report.ok is False
    assert report.error == "budget exceeded"
    assert report.workers == []


def test_report_without_records_falls_back_to_trace_totals():
    report = RunReport.from_outcome(_outcome())
    assert report.trace == {"totals": _outcome().trace_totals}


def test_report_without_run_id_or_totals_has_no_trace():
    outcome = RunOutcome(task="x")
    report = RunReport.from_outcome(outcome, _records())
    assert report.trace is None


def test_report_to_dict_is_json_serializable():
    report = RunReport.from_outcome(_outcome(), _records())
    assert json.loads(json.dumps(report.to_dict()))["run_id"] == "r1"


def test_report_write_json_round_trip(tmp_path):
    report = RunReport.from_outcome(_outcome(), _records())
    path = report.write_json(tmp_path / "reports" / "r1.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["task"] == "search and summarize: quarterly report"
    assert loaded["trace"]["spans"] == 2


def test_run_task_exposes_run_id_matching_its_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    outcome = run_task(
        "search and summarize: quarterly report",
        KeywordRouter({"search": "researcher", "summarize": "summarizer"}),
        [FakeWorker(name="researcher", output="notes"),
         FakeWorker(name="summarizer", output="summary")],
        retry=RetryPolicy(max_attempts=3),
        budget=CostBudget(max_tokens=1000, max_cost_usd=1.0),
        trace_path=str(trace_path),
    )
    assert outcome.ok
    assert outcome.run_id
    records, _ = load_trace_records(trace_path)
    assert all(r["run_id"] == outcome.run_id for r in records)
    report = RunReport.from_outcome(outcome, records)
    assert report.trace is not None
    assert report.trace["spans"] == 2
