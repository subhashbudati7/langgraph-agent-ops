"""Tests for the FastAPI /run endpoint (offline demo stack)."""

import pytest
from fastapi.testclient import TestClient

from agent_ops.api import create_app
from agent_ops.graph import KeywordRouter
from agent_ops.workers import FakeWorker


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_run_returns_worker_results(client):
    response = client.post("/run", json={"task": "search and summarize: report"})
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert [r["worker"] for r in body["results"]] == ["researcher", "summarizer"]


def test_run_rejects_empty_task(client):
    # Pydantic rejects the empty string before the graph ever sees it.
    response = client.post("/run", json={"task": ""})
    assert response.status_code == 422


def test_run_reports_guardrail_violation_not_500(client):
    app = create_app(
        router=KeywordRouter({"summarize": "summarizer"}),
        workers=[FakeWorker(name="summarizer", output="leaked 123-45-6789")],
    )
    response = TestClient(app).post("/run", json={"task": "summarize: report"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert "blocked pattern" in body["error"]


def test_run_reports_budget_breach_not_500(client):
    from agent_ops.policies import CostBudget

    app = create_app(budget=CostBudget(max_tokens=1))
    response = TestClient(app).post("/run", json={"task": "summarize: report"})
    assert response.status_code == 200
    assert "budget" in response.json()["error"]
