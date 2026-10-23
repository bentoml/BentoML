from __future__ import annotations

import asyncio
import typing as t

import numpy as np
import pytest

from bentoml._internal.marshal.dispatcher import CandidateInfo
from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.marshal.dispatcher import Job
from bentoml._internal.marshal.dispatcher import ScheduleDecision
from bentoml._internal.marshal.dispatcher import decide_next_action


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_dispatcher(
    clock: t.Callable[[], float],
    *,
    max_latency_in_ms: int = 100,
    max_batch_size: int = 8,
    get_batch_size: t.Callable[[t.Any], int] = lambda _: 1,
) -> CorkDispatcher[t.Any, t.Any]:
    return CorkDispatcher(
        max_latency_in_ms=max_latency_in_ms,
        max_batch_size=max_batch_size,
        fallback=lambda: "fallback",
        get_batch_size=get_batch_size,
        batch_dim=(0, 0),
        clock=clock,
    )


@pytest.mark.asyncio
async def test_inbound_call_uses_explicit_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)

    task = asyncio.create_task(dispatcher.inbound_call("request", max_latency_ms=25))
    await asyncio.sleep(0)

    job = dispatcher._queue[0]
    assert job.enqueue_time == 100.0
    assert job.deadline == pytest.approx(100.025)

    job.future.set_result("response")
    assert await task == "response"


@pytest.mark.asyncio
async def test_inbound_call_uses_constructor_default() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=80)

    task = asyncio.create_task(dispatcher.inbound_call("request"))
    await asyncio.sleep(0)

    job = dispatcher._queue[0]
    assert job.deadline == pytest.approx(100.08)

    job.future.set_result("response")
    assert await task == "response"


@pytest.mark.asyncio
async def test_static_latency_does_not_build_dynamic_deadline_index() -> None:
    clock = FakeClock()
    dispatcher = CorkDispatcher(
        max_latency_in_ms=80,
        max_batch_size=8,
        fallback=lambda: "fallback",
        get_batch_size=lambda _: 1,
        batch_dim=(0, 0),
        clock=clock,
    )

    task = asyncio.create_task(dispatcher.inbound_call("request"))
    await asyncio.sleep(0)

    job = dispatcher._queue[0]
    assert dispatcher._deadline_heap == []
    assert dispatcher._job_tokens == {}
    assert dispatcher._next_expiry == job.deadline

    job.future.set_result("response")
    assert await task == "response"


@pytest.mark.asyncio
async def test_dispatcher_wrapper_forwards_explicit_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)
    observed: list[int | None] = []

    async def fake_controller() -> None:
        await asyncio.Future()

    async def fake_inbound(data: str, *, max_latency_ms: int | None = None) -> str:
        observed.append(max_latency_ms)
        return data

    monkeypatch.setattr(dispatcher, "controller", fake_controller)
    monkeypatch.setattr(dispatcher, "inbound_call", fake_inbound)

    async def callback(values: t.Sequence[str]) -> t.Sequence[str]:
        return values

    wrapped = dispatcher(callback)
    try:
        assert await wrapped("request", max_latency_ms=12) == "request"
    finally:
        dispatcher.shutdown()

    assert observed == [12]


@pytest.mark.asyncio
async def test_split_job_preserves_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)
    future: asyncio.Future[t.Any | Exception] = (
        asyncio.get_running_loop().create_future()
    )
    job = Job(
        enqueue_time=100.0,
        data=np.array([1, 2, 3, 4]),
        future=future,
        deadline=100.025,
    )

    first, second = dispatcher.split_job(job, [0, 2, 4])

    assert first.deadline == job.deadline
    assert second.deadline == job.deadline


def queued_job(
    dispatcher: CorkDispatcher[t.Any, t.Any],
    *,
    data: tuple[str, int],
    enqueue_time: float,
    deadline: float,
) -> Job[t.Any, t.Any]:
    future: asyncio.Future[t.Any | Exception] = dispatcher._loop.create_future()
    return Job(
        enqueue_time=enqueue_time,
        data=data,
        future=future,
        deadline=deadline,
    )


def test_peek_candidate_matches_batch_size_without_mutating_queue() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(
        clock,
        max_batch_size=10,
        get_batch_size=lambda data: data[1],
    )
    jobs = [
        queued_job(dispatcher, data=("A", 7), enqueue_time=90, deadline=200),
        queued_job(dispatcher, data=("B", 7), enqueue_time=91, deadline=120),
        queued_job(dispatcher, data=("C", 6), enqueue_time=92, deadline=105),
    ]
    for job in jobs:
        dispatcher._append_job(job)

    candidate = dispatcher._peek_candidate()

    assert candidate is not None
    assert candidate.job_count == 2
    assert candidate.sample_size == 10
    assert candidate.earliest_deadline == 120
    assert candidate.last_enqueue_time == 91
    assert list(dispatcher._queue) == jobs


def test_candidate_cache_is_invalidated_by_tighter_enqueue() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=4)
    dispatcher._append_job(
        queued_job(dispatcher, data=("A", 1), enqueue_time=90, deadline=200)
    )
    first = dispatcher._peek_candidate()
    dispatcher._append_job(
        queued_job(dispatcher, data=("B", 1), enqueue_time=91, deadline=110)
    )
    second = dispatcher._peek_candidate()

    assert first is not None
    assert second is not None
    assert first.earliest_deadline == 200
    assert second.earliest_deadline == 110


def test_deadline_outside_candidate_does_not_change_candidate_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=2)
    for name, deadline in [("A", 200), ("B", 210), ("C", 105)]:
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(name, 1),
                enqueue_time=90,
                deadline=deadline,
            )
        )

    candidate = dispatcher._peek_candidate()

    assert candidate is not None
    assert candidate.job_count == 2
    assert candidate.earliest_deadline == 200
    assert dispatcher._next_expiry == 105


def test_expire_jobs_removes_only_expired_jobs_and_preserves_fifo() -> None:
    clock = FakeClock(110)
    dispatcher = make_dispatcher(clock, max_batch_size=2)
    jobs = [
        queued_job(dispatcher, data=("A", 1), enqueue_time=90, deadline=200),
        queued_job(dispatcher, data=("B", 1), enqueue_time=91, deadline=105),
        queued_job(dispatcher, data=("C", 1), enqueue_time=92, deadline=210),
    ]
    for job in jobs:
        dispatcher._append_job(job)

    assert dispatcher._expire_jobs(clock()) == 1
    assert [job.data[0] for job in dispatcher._queue] == ["A", "C"]
    assert jobs[1].future.cancelled()
    assert not jobs[0].future.done()
    assert not jobs[2].future.done()
    assert dispatcher._next_expiry == 200


class CountingDeadline(float):
    comparisons = 0

    def __lt__(self, other: object) -> bool:
        type(self).comparisons += 1
        return super().__lt__(other)


def test_take_inputs_does_not_scan_every_remaining_deadline() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=1)
    for index in range(64):
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(str(index), 1),
                enqueue_time=float(index),
                deadline=CountingDeadline(200 + index),
            )
        )

    CountingDeadline.comparisons = 0
    taken = dispatcher._take_inputs()

    assert len(taken) == 1
    assert CountingDeadline.comparisons < 16
    assert dispatcher._next_expiry == 201


def test_deadline_heap_does_not_accumulate_dispatched_jobs() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_batch_size=2)
    pair_count = 128

    for index in range(pair_count):
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(f"long-{index}", 1),
                enqueue_time=float(index),
                deadline=10_000 + index,
            )
        )
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(f"short-{index}", 1),
                enqueue_time=float(index),
                deadline=200 + index,
            )
        )

    for _ in range(pair_count - 1):
        dispatcher._take_inputs()

    assert len(dispatcher._queue) == 2
    assert len(dispatcher._deadline_heap) <= 2 * len(dispatcher._active_job_tokens)


def test_split_failure_preserves_queue_and_cached_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(
        clock,
        max_batch_size=2,
        get_batch_size=lambda data: data[1],
    )
    job = queued_job(
        dispatcher,
        data=("A", 4),
        enqueue_time=90,
        deadline=105,
    )
    dispatcher._append_job(job)
    candidate = dispatcher._peek_candidate()
    revision = dispatcher._queue_revision

    def fail_split(
        _: Job[t.Any, t.Any], __: list[int]
    ) -> tuple[Job[t.Any, t.Any], Job[t.Any, t.Any]]:
        raise RuntimeError("split failed")

    monkeypatch.setattr(dispatcher, "split_job", fail_split)

    with pytest.raises(RuntimeError, match="split failed"):
        dispatcher._take_inputs()

    assert list(dispatcher._queue) == [job]
    assert dispatcher._queue_revision == revision
    assert dispatcher._peek_candidate() is candidate
    assert dispatcher._next_expiry == 105


def candidate(deadline: float = 110.0) -> CandidateInfo:
    return CandidateInfo(
        job_count=2,
        sample_size=2,
        first_enqueue_time=100.0,
        last_enqueue_time=101.0,
        earliest_deadline=deadline,
    )


def test_decision_waits_when_optimizer_and_deadline_allow() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=False, wait_time=1.0)


def test_decision_caps_wait_at_latest_safe_start() -> None:
    decision = decide_next_action(
        now=107.5,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=False, wait_time=0.5)


def test_decision_dispatches_at_latest_safe_start() -> None:
    decision = decide_next_action(
        now=108.0,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=True, wait_time=0.0)


def test_impossible_deadline_dispatches_when_capacity_is_free() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(deadline=106.0),
        predicted_duration=2.0,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=106.0,
        ticket_interval=1.0,
    )
    assert decision.dispatch


def test_locked_semaphore_waits_until_ticket_or_expiry() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(deadline=106.0),
        predicted_duration=2.0,
        adaptive_should_wait=False,
        semaphore_locked=True,
        next_expiry=105.25,
        ticket_interval=1.0,
    )
    assert decision == ScheduleDecision(dispatch=False, wait_time=0.25)


def test_optimizer_can_release_before_deadline() -> None:
    decision = decide_next_action(
        now=105.0,
        candidate=candidate(),
        predicted_duration=2.0,
        adaptive_should_wait=False,
        semaphore_locked=False,
        next_expiry=110.0,
        ticket_interval=1.0,
    )
    assert decision.dispatch


def test_late_tight_deadline_changes_release_without_reordering() -> None:
    clock = FakeClock(105.0)
    dispatcher = make_dispatcher(clock, max_batch_size=4)
    for name, deadline in [("A", 200.0), ("B", 201.0)]:
        dispatcher._append_job(
            queued_job(
                dispatcher,
                data=(name, 1),
                enqueue_time=100.0,
                deadline=deadline,
            )
        )

    before = dispatcher._peek_candidate()
    assert before is not None
    assert not decide_next_action(
        now=clock(),
        candidate=before,
        predicted_duration=0.002,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=dispatcher._next_expiry,
        ticket_interval=0.001,
    ).dispatch

    dispatcher._append_job(
        queued_job(
            dispatcher,
            data=("C", 1),
            enqueue_time=105.0,
            deadline=105.001,
        )
    )
    after = dispatcher._peek_candidate()
    assert after is not None
    assert decide_next_action(
        now=clock(),
        candidate=after,
        predicted_duration=0.002,
        adaptive_should_wait=True,
        semaphore_locked=False,
        next_expiry=dispatcher._next_expiry,
        ticket_interval=0.001,
    ).dispatch
    assert [job.data[0] for job in dispatcher._take_inputs()] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_expired_job_uses_existing_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=10)

    async def fake_controller() -> None:
        await asyncio.Future()

    monkeypatch.setattr(dispatcher, "controller", fake_controller)

    async def callback(values: t.Sequence[str]) -> t.Sequence[str]:
        return values

    wrapped = dispatcher(callback)
    task = asyncio.create_task(wrapped("expires"))
    await asyncio.sleep(0)
    try:
        clock.advance(0.011)
        assert dispatcher._expire_jobs(clock()) == 1
        assert await task == "fallback"
    finally:
        dispatcher.shutdown()


@pytest.mark.asyncio
async def test_cancelled_future_does_not_shift_other_results() -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock)
    jobs = [
        queued_job(
            dispatcher,
            data=(name, 1),
            enqueue_time=100.0,
            deadline=101.0,
        )
        for name in ("A", "B", "C")
    ]
    for job in jobs:
        dispatcher._append_job(job)

    async def callback(
        values: t.Sequence[tuple[str, int]],
    ) -> t.Sequence[tuple[str, int]]:
        return [(f"{name}-result", size) for name, size in values]

    dispatcher.callback = callback
    assert dispatcher._sema.acquire()
    inputs = dispatcher._take_inputs()
    jobs[1].future.cancel()

    await dispatcher.outbound_call(inputs)

    assert jobs[0].future.result() == ("A-result", 1)
    assert jobs[1].future.cancelled()
    assert jobs[2].future.result() == ("C-result", 1)


@pytest.mark.asyncio
async def test_static_wrapper_keeps_constructor_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=40)

    async def fake_controller() -> None:
        await asyncio.Future()

    monkeypatch.setattr(dispatcher, "controller", fake_controller)

    async def callback(values: t.Sequence[str]) -> t.Sequence[str]:
        return values

    wrapped = dispatcher(callback)
    task = asyncio.create_task(wrapped("legacy"))
    await asyncio.sleep(0)
    try:
        assert dispatcher._queue[0].deadline == pytest.approx(100.04)
        dispatcher._queue[0].future.set_result("legacy")
        assert await task == "legacy"
    finally:
        dispatcher.shutdown()


@pytest.mark.asyncio
async def test_static_controller_skips_dynamic_candidate_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=40)
    candidate_scans = 0
    original_peek_candidate = dispatcher._peek_candidate

    async def skip_training(*_: object) -> None:
        return None

    def count_candidate_scan(*args: object, **kwargs: object) -> CandidateInfo | None:
        nonlocal candidate_scans
        candidate_scans += 1
        return original_peek_candidate(*args, **kwargs)

    monkeypatch.setattr(dispatcher, "train_optimizer", skip_training)
    monkeypatch.setattr(dispatcher, "_peek_candidate", count_candidate_scan)

    async def callback(values: t.Sequence[str]) -> t.Sequence[str]:
        return values

    wrapped = dispatcher(callback)
    try:
        assert await asyncio.wait_for(wrapped("request"), timeout=0.1) == "request"
    finally:
        dispatcher.shutdown()

    assert candidate_scans == 0


@pytest.mark.asyncio
async def test_static_controller_dispatches_predicted_deadline_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    dispatcher = make_dispatcher(clock, max_latency_in_ms=40)
    dispatcher.optimizer.o_a = 0.03
    dispatcher.optimizer.o_b = 0.0
    dispatched: list[list[str]] = []

    async def skip_training(*_: object) -> None:
        return None

    async def callback(
        values: t.Sequence[tuple[str, int]],
    ) -> t.Sequence[tuple[str, int]]:
        dispatched.append([name for name, _ in values])
        return values

    monkeypatch.setattr(dispatcher, "train_optimizer", skip_training)
    dispatcher.callback = callback
    jobs = [
        queued_job(
            dispatcher,
            data=(name, 1),
            enqueue_time=clock(),
            deadline=clock() + 0.04,
        )
        for name in ("A", "B")
    ]
    for job in jobs:
        dispatcher._append_job(job)

    controller = asyncio.create_task(dispatcher.controller())
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(job.future for job in jobs)), timeout=0.1
        )
    finally:
        controller.cancel()
        dispatcher.shutdown()

    assert results == [("A", 1), ("B", 1)]
    assert dispatched == [["A", "B"]]
