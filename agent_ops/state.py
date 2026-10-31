"""Shared graph state for the supervisor/worker workflow."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """State carried through the LangGraph supervisor graph.

    Results are plain dicts (not dataclass instances) so the state stays
    JSON-serializable for checkpointers and trace records.
    """

    task: str
    route: str  # worker name chosen by the supervisor, or "done"
    results: Annotated[list[dict], add]
    error: str | None
