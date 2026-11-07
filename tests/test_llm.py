"""Tests for the OpenAI-compatible LLM client (no network: fake post)."""

from __future__ import annotations

import json

import pytest

from agent_ops.llm import LLMClient, LLMWorker, estimate_tokens
from agent_ops.workers import TransientWorkerError, WorkerError


def _fake_post(body_dict: dict, status: int = 200):
    def post(url: str, headers: dict, body: bytes):
        assert url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer test-key"
        assert json.loads(body)["model"] == "test-model"
        return status, json.dumps(body_dict).encode()

    return post


def test_complete_uses_reported_usage_and_pricing():
    client = LLMClient(
        model="test-model",
        api_key="test-key",
        input_price_per_1k=1.0,
        output_price_per_1k=2.0,
        post=_fake_post(
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ),
    )
    text, tokens, cost = client.complete_with_usage([{"role": "user", "content": "hi"}])
    assert text == "hello"
    assert tokens == 150
    assert cost == pytest.approx(100 * 1.0 / 1000 + 50 * 2.0 / 1000)


def test_complete_falls_back_to_estimate_without_usage():
    client = LLMClient(
        model="test-model",
        api_key="test-key",
        input_price_per_1k=1.0,
        post=_fake_post({"choices": [{"message": {"content": "hello"}}]}),
    )
    text, tokens, cost = client.complete_with_usage([{"role": "user", "content": "hi"}])
    assert text == "hello"
    assert tokens == estimate_tokens("hihello")
    assert cost == pytest.approx(tokens * 1.0 / 1000)


def test_rate_limit_and_server_errors_are_transient():
    for status in (429, 500, 503):
        client = LLMClient(
            model="test-model",
            api_key="test-key",
            post=_fake_post({"error": "boom"}, status=status),
        )
        with pytest.raises(TransientWorkerError):
            client.complete([{"role": "user", "content": "hi"}])


def test_client_error_is_not_retried():
    client = LLMClient(
        model="test-model",
        api_key="test-key",
        post=_fake_post({"error": "bad request"}, status=400),
    )
    with pytest.raises(WorkerError):
        client.complete([{"role": "user", "content": "hi"}])


def test_malformed_response_is_worker_error():
    client = LLMClient(
        model="test-model",
        api_key="test-key",
        post=_fake_post({"choices": []}),
    )
    with pytest.raises(WorkerError):
        client.complete([{"role": "user", "content": "hi"}])


def test_llm_worker_reports_real_tokens_and_cost():
    client = LLMClient(
        model="test-model",
        api_key="test-key",
        input_price_per_1k=1.0,
        output_price_per_1k=2.0,
        post=_fake_post(
            {
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10},
            }
        ),
    )
    worker = LLMWorker(name="summarizer", client=client, system_prompt="Summarize.")
    result = worker.run("quarterly report")
    assert result.worker == "summarizer"
    assert result.output == "done"
    assert result.tokens == 50
    assert result.cost_usd == pytest.approx(40 * 1.0 / 1000 + 10 * 2.0 / 1000)


def test_preset_workers_carry_distinct_roles():
    from agent_ops.llm import (
        critic_worker,
        editor_worker,
        planner_worker,
        researcher_worker,
        summarizer_worker,
    )

    client = LLMClient(
        model="test-model",
        api_key="test-key",
        post=_fake_post({"choices": [{"message": {"content": "ok"}}]}),
    )
    workers = [
        researcher_worker(client),
        summarizer_worker(client),
        critic_worker(client),
        planner_worker(client),
        editor_worker(client),
    ]
    assert [w.name for w in workers] == [
        "researcher",
        "summarizer",
        "critic",
        "planner",
        "editor",
    ]
    prompts = {w.system_prompt for w in workers}
    assert len(prompts) == 5  # each preset has its own system prompt
    for worker in workers:
        assert worker.run("task").output == "ok"
