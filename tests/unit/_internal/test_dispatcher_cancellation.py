import asyncio
import time

import numpy as np
import pytest

from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.marshal.dispatcher import Job


@pytest.mark.asyncio
@pytest.mark.parametrize("training", [False, True])
async def test_cancelled_requests_are_not_dispatched(training):
    batches = []
    dispatcher = CorkDispatcher(
        1000, 2, fallback=lambda: None, get_batch_size=lambda _: 1
    )

    async def callback(inputs):
        batches.append(inputs)
        return inputs

    dispatcher(callback)
    cancelled = asyncio.create_task(dispatcher.inbound_call("cancelled"))
    active = asyncio.create_task(dispatcher.inbound_call("active"))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    dispatcher._sema.acquire()
    await dispatcher.outbound_call(tuple(dispatcher._get_inputs()), training=training)
    assert await active == "active"
    assert batches == [("active",)]
    assert not dispatcher._sema.is_locked()


@pytest.mark.asyncio
async def test_nested_split_checks_parent_before_callbacks_run():
    dispatcher = CorkDispatcher(1000, 2, fallback=lambda: None, get_batch_size=len)
    parent = Job(time.time(), np.arange(5), asyncio.get_running_loop().create_future())
    dispatcher._queue.append(parent)
    tuple(dispatcher._get_inputs())
    tuple(dispatcher._get_inputs())
    parent.future.cancel()
    assert tuple(dispatcher._get_inputs()) == ()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_split_results_remain_in_original_order():
    dispatcher = CorkDispatcher(1000, 2, fallback=lambda: None, get_batch_size=len)
    parent = Job(time.time(), np.arange(5), asyncio.get_running_loop().create_future())
    dispatcher._queue.append(parent)

    async def callback(inputs):
        return tuple(data * 2 for data in inputs)

    dispatcher(callback)
    while dispatcher._queue:
        dispatcher._sema.acquire()
        await dispatcher.outbound_call(tuple(dispatcher._get_inputs()))
        await asyncio.sleep(0)
    result = await asyncio.wait_for(parent.future, timeout=2)
    np.testing.assert_array_equal(result, np.arange(5) * 2)


@pytest.mark.asyncio
async def test_all_cancelled_batch_does_not_call_model():
    dispatcher = CorkDispatcher(
        1000, 2, fallback=lambda: None, get_batch_size=lambda _: 1
    )

    async def callback(inputs):
        pytest.fail("取消的请求不能调用模型")

    dispatcher(callback)
    future = asyncio.get_running_loop().create_future()
    job = Job(time.time(), "cancelled", future)
    future.cancel()
    dispatcher._queue.append(job)
    dispatcher._sema.acquire()
    await dispatcher.outbound_call(tuple(dispatcher._get_inputs()))
    assert not dispatcher._sema.is_locked()


@pytest.mark.asyncio
async def test_cancelling_split_parent_skips_remaining_child():
    dispatcher = CorkDispatcher(1000, 2, fallback=lambda: None, get_batch_size=len)
    parent = Job(
        time.time(), np.array([1, 2, 3]), asyncio.get_running_loop().create_future()
    )
    dispatcher._queue.append(parent)
    first = tuple(dispatcher._get_inputs())
    assert len(first) == 1
    assert len(dispatcher._queue) == 1
    parent.future.cancel()
    await asyncio.sleep(0)
    assert first[0].future.cancelled()
    assert tuple(dispatcher._get_inputs()) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("training", [False, True])
async def test_controller_skips_cancelled_queue_and_serves_next_request(training):
    dispatcher = CorkDispatcher(
        1000, 2, fallback=lambda: None, get_batch_size=lambda _: 1
    )
    batches = []

    async def callback(inputs):
        batches.append(inputs)
        return inputs

    dispatcher(callback)
    cancelled = asyncio.create_task(dispatcher.inbound_call("cancelled"))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    if training:
        controller = asyncio.create_task(dispatcher.train_optimizer(1, 1, 1))
    else:

        async def skip_training(*args):
            pass

        dispatcher.train_optimizer = skip_training
        dispatcher.optimizer.o_a = 0
        dispatcher.optimizer.o_b = 0
        dispatcher.optimizer.wait = 0
        controller = asyncio.create_task(dispatcher.controller())
    try:
        await asyncio.sleep(0.01)
        assert batches == []
        assert not controller.done()
        result = await asyncio.wait_for(dispatcher.inbound_call("active"), timeout=2)
        assert result == "active"
        assert batches == [("active",)]
        assert not dispatcher._sema.is_locked()
    finally:
        controller.cancel()
        await asyncio.gather(controller, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_inflight_request_preserves_other_batch_results():
    dispatcher = CorkDispatcher(
        1000, 2, fallback=lambda: None, get_batch_size=lambda _: 1
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def callback(inputs):
        entered.set()
        await release.wait()
        return inputs

    dispatcher(callback)
    first = asyncio.create_task(dispatcher.inbound_call("first"))
    second = asyncio.create_task(dispatcher.inbound_call("second"))
    await asyncio.sleep(0)
    dispatcher._sema.acquire()
    outbound = asyncio.create_task(
        dispatcher.outbound_call(tuple(dispatcher._get_inputs()))
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not outbound.cancelled()
    release.set()
    await outbound
    assert await second == "second"
    assert not dispatcher._sema.is_locked()
