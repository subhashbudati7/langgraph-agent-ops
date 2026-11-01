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
