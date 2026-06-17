from __future__ import annotations

import asyncio

import pytest

from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.marshal.dispatcher import Job


def test_dispatch_observer_receives_queue_and_batch_shape() -> None:
    loop = asyncio.new_event_loop()
    try:
        observed = []
        dispatcher = CorkDispatcher(
            max_latency_in_ms=100,
            max_batch_size=8,
            fallback=lambda: None,
            get_batch_size=len,
            dispatch_observer=observed.append,
        )
        jobs = (
            Job(10.0, [1, 2, 3], loop.create_future()),
            Job(12.0, [4, 5], loop.create_future()),
        )

        dispatcher._observe_dispatch(
            jobs,
            reason="max_batch_size",
            training=False,
            dispatch_time=15.0,
        )

        assert len(observed) == 1
        metrics = observed[0]
        assert metrics.reason == "max_batch_size"
        assert metrics.training is False
        assert metrics.queue_size == 2
        assert metrics.job_count == 2
        assert metrics.item_count == 5
        assert metrics.max_batch_size == 8
        assert metrics.oldest_queue_wait_seconds == 5.0
        assert metrics.newest_queue_wait_seconds == 3.0
        assert metrics.request_batch_sizes == (3, 2)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_dispatch_observer_runs_on_real_dispatch() -> None:
    observed = []
    dispatcher = CorkDispatcher(
        max_latency_in_ms=100,
        max_batch_size=4,
        fallback=lambda: None,
        get_batch_size=len,
        dispatch_observer=observed.append,
    )

    async def callback(items: tuple[list[int], ...]) -> list[str]:
        return [str(sum(item)) for item in items]

    dispatcher.callback = callback
    loop = asyncio.get_running_loop()
    jobs = (
        Job(loop.time(), [1, 2], loop.create_future()),
        Job(loop.time(), [3], loop.create_future()),
    )

    dispatcher._dispatch(jobs, reason="optimizer")

    assert await jobs[0].future == "3"
    assert await jobs[1].future == "3"
    assert jobs[0].dispatch_time > 0
    assert jobs[1].dispatch_time == jobs[0].dispatch_time
    assert observed[0].reason == "optimizer"
    assert observed[0].item_count == 3


def test_dispatch_observer_error_does_not_break_dispatch_metrics() -> None:
    loop = asyncio.new_event_loop()
    try:
        dispatcher = CorkDispatcher(
            max_latency_in_ms=100,
            max_batch_size=4,
            fallback=lambda: None,
            get_batch_size=len,
            dispatch_observer=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        jobs = (Job(10.0, [1], loop.create_future()),)

        dispatcher._observe_dispatch(
            jobs,
            reason="optimizer",
            training=False,
            dispatch_time=11.0,
        )
    finally:
        loop.close()
