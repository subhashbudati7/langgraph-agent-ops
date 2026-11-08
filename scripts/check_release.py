#!/usr/bin/env python3
"""One-command pre-merge verification for the November release.

Runs, in order: ruff, pytest, the offline demo, the trace summarizer
over the demo trace, and the run-report script. Prints PASS/FAIL per
step and exits non-zero if any step fails, so CI or a deploy run can
treat it as a gate. Uses the current interpreter (`sys.executable`)
so it runs against the project venv when invoked from it.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = sys.executable


def _steps(tmp_dir: str) -> list[tuple[str, list[str]]]:
    trace_path = str(Path(tmp_dir) / "check-trace.jsonl")
    return [
        ("ruff", [VENV_PYTHON, "-m", "ruff", "check", "."]),
        ("pytest", [VENV_PYTHON, "-m", "pytest", "-q"]),
        ("offline demo", [VENV_PYTHON, "scripts/demo.py",
                          "--trace-path", trace_path]),
        ("trace summary", [VENV_PYTHON, "scripts/summarize_trace.py",
                           "--trace-path", trace_path]),
        ("run report", [VENV_PYTHON, "scripts/run_report.py",
                        "--trace-path", trace_path,
                        "--report-dir", str(Path(tmp_dir) / "reports")]),
    ]


def run_step(
    name: str,
    cmd: list[str],
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Run one step, print PASS/FAIL, return True on success."""
    result = runner(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"PASS {name}")
        return True
    print(f"FAIL {name} (exit {result.returncode})")
    tail = (result.stdout or "") + (result.stderr or "")
    for line in tail.strip().splitlines()[-5:]:
        print(f"  {line}")
    return False


def main(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    with tempfile.TemporaryDirectory(prefix="agent-ops-check-") as tmp_dir:
        results = [run_step(name, cmd, runner) for name, cmd in _steps(tmp_dir)]
    if all(results):
        print("check_release: all steps PASS")
        return 0
    failed = sum(1 for ok in results if not ok)
    print(f"check_release: {failed} step(s) FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
