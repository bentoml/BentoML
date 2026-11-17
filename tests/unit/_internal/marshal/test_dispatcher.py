from __future__ import annotations

import asyncio

import numpy as np
import pytest

from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.marshal.dispatcher import Job


@pytest.fixture
def dispatcher():
    instance = CorkDispatcher(
        max_latency_in_ms=1000,
        max_batch_size=4,
        fallback=lambda: None,
        get_batch_size=len,
    )
    yield instance
    instance.shutdown()


@pytest.fixture
def loop_errors(event_loop):
    errors = []
    previous = event_loop.get_exception_handler()
    event_loop.set_exception_handler(lambda _loop, context: errors.append(context))
    yield errors
    event_loop.set_exception_handler(previous)


def split(dispatcher):
    job = Job(0, np.arange(6), asyncio.get_running_loop().create_future())
    children = dispatcher.split_job(job, [0, 4, 6])
    return job, children


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
async def test_split_job_propagates_recombination_error(dispatcher, loop_errors, order):
    job, children = split(dispatcher)
    outputs = [np.zeros((4, 4)), np.zeros((2, 6))]
    for index in order:
        children[index].future.set_result(outputs[index])
    await asyncio.sleep(0)

    assert job.future.done()
    assert isinstance(job.future.result(), ValueError)
    assert loop_errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
async def test_split_job_combines_compatible_results(dispatcher, loop_errors, order):
    job, children = split(dispatcher)
    outputs = [np.arange(8).reshape(4, 2), np.arange(8, 12).reshape(2, 2)]
    for index in order:
        children[index].future.set_result(outputs[index])
    await asyncio.sleep(0)

    np.testing.assert_array_equal(job.future.result(), np.arange(12).reshape(6, 2))
    assert loop_errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("failed_child", [0, 1])
async def test_split_job_preserves_child_error(
    dispatcher, loop_errors, order, failed_child
):
    job, children = split(dispatcher)
    error = RuntimeError("inference failed")
    outputs = [np.zeros((4, 2)), np.zeros((2, 2))]
    outputs[failed_child] = error
    for index in order:
        children[index].future.set_result(outputs[index])
    await asyncio.sleep(0)

    assert job.future.result() is error
    assert loop_errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("cancelled_child", [0, 1])
async def test_split_job_propagates_child_cancellation(
    dispatcher, loop_errors, order, cancelled_child
):
    job, children = split(dispatcher)
    for index in order:
        if index == cancelled_child:
            children[index].future.cancel()
        else:
            children[index].future.set_result(np.zeros((len(children[index].data), 2)))
    await asyncio.sleep(0)

    assert job.future.cancelled()
    assert loop_errors == []


@pytest.mark.asyncio
async def test_split_job_keeps_cancelled_parent(dispatcher, loop_errors):
    job, children = split(dispatcher)
    job.future.cancel()
    for child in children:
        child.future.set_result(np.zeros((len(child.data), 2)))
    await asyncio.sleep(0)

    assert job.future.cancelled()
    assert loop_errors == []
