"""JSONL tracing: one record per node invocation, replayable later."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class TraceRecorder:
    """Appends span records to a JSONL file as the run progresses."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.records: list[dict] = []

    @contextmanager
    def span(self, node: str) -> Iterator[dict]:
        """Record one node's invocation; the yielded dict is filled in place."""
        record: dict = {"run_id": self.run_id, "node": node, "status": "ok"}
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
            self.records.append(record)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def totals(self) -> dict:
        """Aggregate tokens, cost, and latency across recorded spans."""
        return {
            "spans": len(self.records),
            "tokens": sum(r.get("tokens", 0) for r in self.records),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in self.records), 6),
            "latency_ms": round(sum(r.get("latency_ms", 0.0) for r in self.records), 3),
            "errors": sum(1 for r in self.records if r["status"] == "error"),
        }
