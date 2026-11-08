"""Tests for the pre-merge release check (scripts/check_release.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_release


def _ok(cmd, **kwargs):
    return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")


def _fail(cmd, **kwargs):
    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")


def test_run_step_passes_on_zero_exit(capsys):
    assert check_release.run_step("ruff", ["ruff"], _ok) is True
    assert "PASS ruff" in capsys.readouterr().out


def test_run_step_fails_and_shows_tail(capsys):
    assert check_release.run_step("pytest", ["pytest"], _fail) is False
    out = capsys.readouterr().out
    assert "FAIL pytest (exit 1)" in out
    assert "boom" in out


def test_main_all_pass(capsys):
    assert check_release.main(_ok) == 0
    assert "all steps PASS" in capsys.readouterr().out


def test_main_one_failure_exits_nonzero(capsys):
    calls = {"n": 0}

    def flaky(cmd, **kwargs):
        calls["n"] += 1
        return _fail(cmd) if calls["n"] == 2 else _ok(cmd)

    assert check_release.main(flaky) == 1
    assert "1 step(s) FAILED" in capsys.readouterr().out


def test_steps_cover_lint_tests_demo_summary_report():
    names = [name for name, _ in check_release._steps("/tmp/x")]
    assert names == ["ruff", "pytest", "offline demo", "trace summary", "run report"]
