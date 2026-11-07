"""OpenAI-compatible LLM client for real worker agents.

Uses only the standard library for HTTP, so the package gains no new
runtime dependency. The client is deliberately small: one chat-completion
call, usage accounting, and error mapping onto the worker error types
so the retry policy treats rate limits and 5xx as transient.

A `post` callable can be injected for tests (and for users who want
their own transport); the default transport posts JSON with urllib.
No API key is ever logged: error messages carry the HTTP status only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_ops.workers import TransientWorkerError, WorkerError, WorkerResult

PostFn = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for budgeting before a call."""
    return max(1, len(text) // 4)


def _default_post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise TransientWorkerError(f"LLM request failed: {exc.reason}") from exc


@dataclass
class LLMClient:
    """Minimal OpenAI-compatible chat client.

    `input_price_per_1k` / `output_price_per_1k` are USD per 1,000
    tokens; set them from the model provider's pricing page. When the
    API reports usage, real token counts are used; otherwise the
    estimate above fills in, so cost accounting never silently drops
    to zero.
    """

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0
    post: PostFn = field(default=_default_post, repr=False)

    def complete_with_usage(self, messages: list[dict]) -> tuple[str, int, float]:
        """Run one chat completion; return (text, total_tokens, cost_usd)."""
        payload = json.dumps({"model": self.model, "messages": messages}).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        status, body = self.post(f"{self.base_url}/chat/completions", headers, payload)
        if status == 429 or status >= 500:
            raise TransientWorkerError(f"LLM request failed with HTTP {status}")
        if status != 200:
            raise WorkerError(f"LLM request failed with HTTP {status}")
        try:
            data = json.loads(body)
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise WorkerError("LLM response was not a chat completion") from exc
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            tokens = estimate_tokens(" ".join(m["content"] for m in messages) + text)
            cost = tokens * (self.input_price_per_1k / 1000)
            return text, tokens, cost
        tokens = prompt_tokens + completion_tokens
        cost = (
            prompt_tokens * self.input_price_per_1k / 1000
            + completion_tokens * self.output_price_per_1k / 1000
        )
        return text, tokens, cost

    def complete(self, messages: list[dict]) -> str:
        """Run one chat completion and return only the text."""
        text, _, _ = self.complete_with_usage(messages)
        return text


@dataclass
class LLMWorker:
    """A worker agent backed by an LLM client.

    Sends a system prompt plus the task as the user message, and
    reports the client's real token/cost accounting so the cost
    budget and trace totals reflect LLM spend, not estimates.
    """

    name: str
    client: LLMClient
    system_prompt: str

    def run(self, task: str) -> WorkerResult:
        text, tokens, cost = self.client.complete_with_usage(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task},
            ]
        )
        return WorkerResult(worker=self.name, output=text, tokens=tokens, cost_usd=cost)


def researcher_worker(client: LLMClient) -> LLMWorker:
    """Gather facts relevant to the task; report findings, not conclusions."""
    return LLMWorker(
        name="researcher",
        client=client,
        system_prompt=(
            "You are a researcher. Gather the key facts relevant to the task "
            "and report them as concise bullet points with no conclusions."
        ),
    )


def summarizer_worker(client: LLMClient) -> LLMWorker:
    """Condense prior findings into a short summary for the requester."""
    return LLMWorker(
        name="summarizer",
        client=client,
        system_prompt=(
            "You are a summarizer. Condense the task and any findings into a "
            "short summary a busy reader can act on."
        ),
    )


def critic_worker(client: LLMClient) -> LLMWorker:
    """Check a draft for gaps, unsupported claims, and missing steps."""
    return LLMWorker(
        name="critic",
        client=client,
        system_prompt=(
            "You are a critic. Review the task output for gaps, unsupported "
            "claims, and missing steps, and list each issue briefly."
        ),
    )


def planner_worker(client: LLMClient) -> LLMWorker:
    """Break a task into ordered steps before any worker executes them.

    Runs first in a plan-then-execute pipeline: the supervisor routes
    to the planner, and downstream workers use its step list as the
    plan of record instead of re-deriving one from the raw task.
    """
    return LLMWorker(
        name="planner",
        client=client,
        system_prompt=(
            "You are a planner. Break the task into a short numbered list "
            "of concrete steps, in execution order, with nothing else. "
            "Do not execute the steps yourself."
        ),
    )


def editor_worker(client: LLMClient) -> LLMWorker:
    """Polish prior worker outputs into one final deliverable.

    Runs last: it receives the task plus the findings so far (via the
    supervisor's results) and returns a clean, self-contained answer
    with no loose ends or duplicated sections.
    """
    return LLMWorker(
        name="editor",
        client=client,
        system_prompt=(
            "You are an editor. Combine the task and the prior findings "
            "into one clean, self-contained answer: remove duplication, "
            "fill obvious gaps, and keep only what the requester needs."
        ),
    )
