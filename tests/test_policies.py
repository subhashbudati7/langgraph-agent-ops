import pytest

from agent_ops.policies import BudgetExceeded, CostBudget, RetryPolicy, run_worker_with_policy
from agent_ops.workers import FakeWorker, WorkerError


def test_retry_succeeds_after_transient_failures():
    worker = FakeWorker(name="w", fail_times=2)
    sleeps: list[float] = []
    result = run_worker_with_policy(
        worker, "task", retry=RetryPolicy(max_attempts=3, base_delay_s=1.0, factor=2.0),
        sleep=sleeps.append,
    )
    assert result.worker == "w"
    assert worker.calls == 3
    assert sleeps == [1.0, 2.0]


def test_retry_gives_up_after_max_attempts():
    worker = FakeWorker(name="w", fail_times=5)
    with pytest.raises(Exception, match="transient failure"):
        run_worker_with_policy(worker, "task", retry=RetryPolicy(max_attempts=2), sleep=lambda _: None)
    assert worker.calls == 2


def test_non_transient_error_not_retried():
    class BadWorker:
        name = "bad"

        def run(self, task):
            raise WorkerError("permanent")

    worker = BadWorker()
    with pytest.raises(WorkerError, match="permanent"):
        run_worker_with_policy(worker, "task", sleep=lambda _: None)


def test_backoff_delays_capped():
    policy = RetryPolicy(max_attempts=5, base_delay_s=10.0, factor=3.0, max_delay_s=25.0)
    assert policy.delays() == [10.0, 25.0, 25.0, 25.0]


def test_budget_records_and_blocks_overrun():
    budget = CostBudget(max_tokens=15, max_cost_usd=0.01)
    cheap = FakeWorker(name="cheap", tokens=10, cost_usd=0.002)
    run_worker_with_policy(cheap, "task", budget=budget, sleep=lambda _: None)
    assert budget.tokens_used == 10
    pricey = FakeWorker(name="pricey", tokens=10, cost_usd=0.002)
    with pytest.raises(BudgetExceeded, match="token budget"):
        run_worker_with_policy(pricey, "task", budget=budget, sleep=lambda _: None)
    # Failed budget check does not record the rejected result.
    assert budget.tokens_used == 10


def test_budget_cost_cap():
    budget = CostBudget(max_cost_usd=0.001)
    worker = FakeWorker(name="w", tokens=1, cost_usd=0.002)
    with pytest.raises(BudgetExceeded, match="cost budget"):
        run_worker_with_policy(worker, "task", budget=budget, sleep=lambda _: None)


def test_precheck_fails_fast_without_calling_worker():
    from agent_ops.policies import BudgetExceeded, CostBudget, run_worker_with_policy
    from agent_ops.workers import FakeWorker

    worker = FakeWorker(name="w", tokens=10)
    budget = CostBudget(max_tokens=20)
    with pytest.raises(BudgetExceeded):
        run_worker_with_policy(worker, "task", budget=budget, estimate_tokens=lambda _: 50)
    assert worker.calls == 0
    assert budget.tokens_used == 0


def test_precheck_passes_then_postcheck_records():
    from agent_ops.policies import CostBudget, run_worker_with_policy
    from agent_ops.workers import FakeWorker

    worker = FakeWorker(name="w", tokens=10)
    budget = CostBudget(max_tokens=20)
    result = run_worker_with_policy(
        worker, "task", budget=budget, estimate_tokens=lambda _: 5
    )
    assert result.tokens == 10
    assert budget.tokens_used == 10
