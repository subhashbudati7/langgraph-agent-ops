from agent_ops.graph import KeywordRouter, build_graph
from agent_ops.workers import FakeWorker


def _run(task, workers, router):
    app = build_graph(router, workers)
    return app.invoke({"task": task, "route": "", "results": [], "error": None})


def test_single_worker_route_finishes():
    worker = FakeWorker(name="summarizer", output="summary here")
    final = _run(
        "summarize: quarterly report",
        [worker],
        KeywordRouter({"summarize": "summarizer"}),
    )
    assert final["route"] == "done"
    assert len(final["results"]) == 1
    assert final["results"][0]["output"] == "summary here"
    assert worker.calls == 1


def test_multi_worker_task_routes_to_each_once():
    workers = [FakeWorker(name="researcher"), FakeWorker(name="summarizer")]
    final = _run(
        "search and summarize: quarterly report",
        workers,
        KeywordRouter({"search": "researcher", "summarize": "summarizer"}),
    )
    assert [r["worker"] for r in final["results"]] == ["researcher", "summarizer"]
    assert final["route"] == "done"


def test_unmatched_task_with_no_default_finishes_empty():
    final = _run(
        "do something unknown",
        [FakeWorker(name="summarizer")],
        KeywordRouter({"summarize": "summarizer"}),
    )
    assert final["route"] == "done"
    assert final["results"] == []


def test_default_worker_used_for_unmatched_task():
    worker = FakeWorker(name="generalist")
    final = _run(
        "do something unknown",
        [worker],
        KeywordRouter({}, default="generalist"),
    )
    assert final["results"][0]["worker"] == "generalist"
    assert worker.calls == 1


def test_duplicate_worker_names_rejected():
    try:
        build_graph(
            KeywordRouter({}),
            [FakeWorker(name="dup"), FakeWorker(name="dup")],
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_llm_router_routes_choice_and_done():
    from agent_ops.graph import LLMRouter

    router = LLMRouter(["researcher", "summarizer"], lambda _: "researcher")
    assert router.route("search and summarize", []) == "researcher"
    assert router.route("search and summarize", [{"worker": "researcher"}]) == "done"


def test_llm_router_unknown_or_repeat_choice_ends_run():
    from agent_ops.graph import LLMRouter

    router = LLMRouter(["researcher"], lambda _: "nobody")
    assert router.route("task", []) == "done"
    router = LLMRouter(["researcher"], lambda _: "researcher")
    assert router.route("task", [{"worker": "researcher"}]) == "done"


class _FlakyRouter:
    """Fails `fail_times` times with a transient error, then routes."""

    def __init__(self, fail_times: int, choice: str = "researcher") -> None:
        self.fail_times = fail_times
        self.choice = choice
        self.calls = 0

    def route(self, task: str, results: list[dict]) -> str:
        from agent_ops.workers import TransientWorkerError

        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientWorkerError("router 429")
        if results:
            return "done"
        return self.choice


def test_retrying_router_recovers_after_transient_failures():
    from agent_ops.graph import RetryingRouter

    sleeps: list[float] = []
    router = RetryingRouter(
        _FlakyRouter(fail_times=2), max_attempts=3, sleep=sleeps.append
    )
    assert router.route("task", []) == "researcher"
    assert sleeps == [1.0, 2.0]  # exponential backoff between the 3 attempts


def test_retrying_router_falls_back_when_all_attempts_fail():
    from agent_ops.graph import KeywordRouter, RetryingRouter

    fallback = KeywordRouter({"summarize": "summarizer"})
    router = RetryingRouter(
        _FlakyRouter(fail_times=99),
        fallback=fallback,
        max_attempts=2,
        sleep=lambda _: None,
    )
    assert router.route("summarize this", []) == "summarizer"


def test_retrying_router_defaults_to_done_when_all_attempts_fail():
    from agent_ops.graph import RetryingRouter

    router = RetryingRouter(
        _FlakyRouter(fail_times=99), max_attempts=2, sleep=lambda _: None
    )
    assert router.route("task", []) == "done"


def test_retrying_router_does_not_retry_non_transient_errors():
    from agent_ops.graph import RetryingRouter

    class _BrokenRouter:
        def route(self, task: str, results: list[dict]) -> str:
            raise ValueError("misconfigured router")

    router = RetryingRouter(_BrokenRouter(), max_attempts=3, sleep=lambda _: None)
    try:
        router.route("task", [])
    except ValueError as exc:
        assert "misconfigured" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_retrying_router_in_graph_survives_flaky_llm_router():
    from agent_ops.graph import RetryingRouter

    worker = FakeWorker(name="researcher", output="findings")
    router = RetryingRouter(
        _FlakyRouter(fail_times=1), max_attempts=3, sleep=lambda _: None
    )
    final = _run("research this", [worker], router)
    assert [r["worker"] for r in final["results"]] == ["researcher"]
    assert final["route"] == "done"
