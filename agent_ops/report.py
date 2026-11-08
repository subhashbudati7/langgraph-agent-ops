"""Run reports: one JSON-serializable record per task run.

A `RunReport` ties a `RunOutcome` to its trace summary (per-node
spans, tokens, cost, latency, errors) so a run can be reviewed,
compared, or archived without re-reading the raw JSONL trace.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent_ops.runner import RunOutcome
from agent_ops.summary import summarize_trace


@dataclass
class RunReport:
    run_id: str | None
    task: str
    ok: bool
    error: str | None
    workers: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    trace: dict | None = None
    generated_at: str = ""

    @classmethod
    def from_outcome(
        cls, outcome: RunOutcome, records: list[dict] | None = None
    ) -> RunReport:
        """Build a report from an outcome, optionally with its trace records.

        When records are given and the outcome has a run_id, only that
        run's summary is attached; without records (or without a
        run_id) the report carries the outcome's trace totals instead,
        and `trace` stays None when there is nothing to attach.
        """
        trace = None
        if records is not None and outcome.run_id is not None:
            summary = summarize_trace(records)
            trace = summary["runs"].get(outcome.run_id)
        elif outcome.trace_totals is not None:
            trace = {"totals": outcome.trace_totals}
        return cls(
            run_id=outcome.run_id,
            task=outcome.task,
            ok=outcome.ok,
            error=outcome.error,
            workers=[r.get("worker", "") for r in outcome.results],
            results=list(outcome.results),
            trace=trace,
            generated_at=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict:
        """Plain-dict form, safe to hand to `json.dumps`."""
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        """Write the report as indented JSON; returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path
