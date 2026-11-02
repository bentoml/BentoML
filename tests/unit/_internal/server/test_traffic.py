from __future__ import annotations

import asyncio
import json
import typing as t

import pytest

from bentoml._internal.server.http.traffic import TimeoutMiddleware

if t.TYPE_CHECKING:
    from bentoml._internal import external_typing as ext

SCOPE = {"type": "http", "method": "GET", "path": "/", "headers": []}


async def receive() -> ext.ASGIMessage:
    return {"type": "http.request", "body": b"", "more_body": False}


async def ok_app(
    scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def hanging_app(
    scope: ext.ASGIScope, receive: ext.ASGIReceive, send: ext.ASGISend
) -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_timeout_middleware_cancels_timer(monkeypatch: pytest.MonkeyPatch):
    loop = asyncio.get_running_loop()
    handles: list[asyncio.TimerHandle] = []
    call_later = loop.call_later

    def _call_later(*args: t.Any, **kwargs: t.Any) -> asyncio.TimerHandle:
        handle = call_later(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(loop, "call_later", _call_later)

    middleware = TimeoutMiddleware(ok_app, timeout=60)
    for _ in range(10):
        messages: list[ext.ASGIMessage] = []

        async def send(message: ext.ASGIMessage) -> None:
            messages.append(message)

        await middleware(SCOPE, receive, send)
        assert messages[0]["status"] == 200

    # every per-request timer must be released once its request has finished,
    # otherwise each one keeps the request alive for the whole timeout
    assert len(handles) == 10
    assert all(handle.cancelled() for handle in handles)


@pytest.mark.asyncio
async def test_timeout_middleware_returns_504_on_timeout():
    middleware = TimeoutMiddleware(hanging_app, timeout=0.1)
    messages: list[ext.ASGIMessage] = []

    async def send(message: ext.ASGIMessage) -> None:
        messages.append(message)

    await asyncio.wait_for(middleware(SCOPE, receive, send), timeout=5)

    assert messages[0]["status"] == 504
    assert json.loads(messages[1]["body"]) == {
        "error": "Not able to process the request in 0.1 seconds"
    }
