"""Input/output guardrails for tasks and worker outputs."""

from __future__ import annotations

import re

BLOCKED_PATTERNS = (
    re.compile(r"ignore (all )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (all )?prior instructions", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-shaped numbers: never pass through
)

# Outputs get the PII scan plus secret shapes: an LLM that echoes a key
# from its context (or hallucinates a plausible one) must not have that
# output stored in state, traced, or returned to the caller.
OUTPUT_BLOCKED_PATTERNS = (
    BLOCKED_PATTERNS[2],
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),  # OpenAI-style API key
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
)


class GuardrailViolation(Exception):
    """Raised when a task or output fails a guardrail check."""


def check_input(task: str, max_length: int = 4000) -> None:
    """Validate a task before it enters the graph."""
    if not task.strip():
        raise GuardrailViolation("task is empty")
    if len(task) > max_length:
        raise GuardrailViolation(f"task exceeds {max_length} characters")
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(task):
            raise GuardrailViolation(f"task matches blocked pattern: {pattern.pattern}")


def check_output(output: str, max_length: int = 20000) -> None:
    """Validate a worker output before it is returned to the caller."""
    if len(output) > max_length:
        raise GuardrailViolation(f"output exceeds {max_length} characters")
    for pattern in OUTPUT_BLOCKED_PATTERNS:  # outputs: PII + secret scan
        if pattern.search(output):
            raise GuardrailViolation(f"output matches blocked pattern: {pattern.pattern}")
