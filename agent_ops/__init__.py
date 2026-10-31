"""Multi-agent workflow ops on LangGraph.

Supervisor routes tasks to worker agents; every worker call runs under a
retry/backoff policy, a cost budget, input/output guardrails, and a JSONL
trace. The graph runs offline with deterministic fake workers; a real LLM
worker is a later, separately reviewed addition.
"""

__version__ = "0.1.0"
