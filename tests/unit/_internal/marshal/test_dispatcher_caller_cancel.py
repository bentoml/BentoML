from __future__ import annotations

import asyncio
import typing as t

import pytest
from starlette.responses import JSONResponse

from bentoml._internal.marshal.dispatcher import CorkDispatcher
from bentoml._internal.marshal.dispatcher import NonBlockSema
from bentoml._internal.server.http.traffic import TimeoutMiddleware
from bentoml.exceptions import ServiceUnavailable

if t.TYPE_CHECKING:
    from bentoml._internal import external_typing as ext

FALLBACK = "fallback"


async def echo(inputs: t.Sequence[str]) -> t.Sequence[str]:
    return inputs


def held_sema() -> NonBlockSema:
    """A batch slot that is already taken, so new requests stay in the queue."""
    sema = NonBlockSema(1)
    sema.acquire()
    return sema


def make_dispatcher(
    max_latency_in_ms: int = 60_000, fallback: t.Callable[[], t.Any] = lambda: FALLBACK
) -> CorkDispatcher[str, str]:
    return CorkDispatcher(
        max_latency_in_ms,
        4,
        shared_sema=held_sema(),
        fallback=fallback,
        get_batch_size=lambda _: 1,
    )


async def wait_queued(dispatcher: CorkDispatcher[str, str], n: int) -> None:
    async def poll() -> None:
        while len(dispatcher._queue) < n:  # type: ignore
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=5)


@pytest.mark.asyncio
async def test_cancelled_caller_gets_cancelled_error() -> None:
    dispatcher = make_dispatcher()
    task = asyncio.ensure_future(dispatcher(echo)("a"))
    try:
        await wait_queued(dispatcher, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        dispatcher.shutdown()


@pytest.mark.asyncio
async def test_timed_out_caller_gets_timeout_error() -> None:
    dispatcher = make_dispatcher()
    func = dispatcher(echo)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(func("a"), timeout=0.01)
    finally:
        dispatcher.shutdown()


@pytest.mark.asyncio
async def test_job_shed_by_the_controller_gets_the_fallback() -> None:
    # The held semaphore keeps the controller in its warm-up phase, which
    # sheds only while more than two jobs are queued. With max_latency_in_ms=1
    # it drops the oldest of three once that job has waited 1 ms.
    dispatcher = make_dispatcher(max_latency_in_ms=1)
    func = dispatcher(echo)
    oldest = asyncio.ensure_future(func("a"))
    others = [asyncio.ensure_future(func(x)) for x in "bc"]
    try:
        assert await asyncio.wait_for(oldest, timeout=5) == FALLBACK
        assert not any(task.done() for task in others)
    finally:
        dispatcher.shutdown()
        await asyncio.gather(*others, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_gives_queued_callers_the_fallback() -> None:
    dispatcher = make_dispatcher()
    tasks = [asyncio.ensure_future(dispatcher(echo)(x)) for x in "ab"]
    await wait_queued(dispatcher, 2)
    dispatcher.shutdown()
    assert await asyncio.wait_for(asyncio.gather(*tasks), timeout=5) == [
        FALLBACK,
        FALLBACK,
    ]


@pytest.mark.asyncio
async def test_timed_out_request_is_answered_once() -> None:
    def overloaded() -> t.NoReturn:
        raise ServiceUnavailable("process is overloaded")

    dispatcher = make_dispatcher(fallback=overloaded)
    func = dispatcher(echo)
    finished = asyncio.Event()

    async def endpoint(
        scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
    ) -> None:
        # like the service app: an overload error is answered with a 503
        try:
            await func("a")
        except ServiceUnavailable:
            await JSONResponse({}, status_code=503)(scope, receive, send)
        finally:
            finished.set()

    statuses: list[int] = []

    async def send(message: ext.ASGIMessage) -> None:
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    async def receive() -> ext.ASGIMessage:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope: ext.ASGIScope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
    }
    try:
        await TimeoutMiddleware(endpoint, timeout=0.05)(scope, receive, send)
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert statuses == [504]
    finally:
        dispatcher.shutdown()
