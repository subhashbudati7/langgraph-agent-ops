"""Multi-agent workflow ops on LangGraph.

Supervisor routes tasks to worker agents; every worker call runs under a
retry/backoff policy, a cost budget, input/output guardrails, and a JSONL
trace. The graph runs offline with deterministic fake workers, and
`agent_ops.llm` adds real OpenAI-compatible LLM workers, an LLM router,
and researcher/summarizer/critic presets behind the same ops layer.
"""

__version__ = "0.1.0"
