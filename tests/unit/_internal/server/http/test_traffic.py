from __future__ import annotations

import asyncio
import time

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from bentoml._internal.server.http.traffic import TimeoutMiddleware


async def _asgi_call(app, path: str) -> tuple[float, list[dict], BaseException | None]:
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 123),
        "server": ("127.0.0.1", 80),
    }
    started = time.perf_counter()
    error: BaseException | None = None
    try:
        await app(scope, receive, send)
    except BaseException as exc:
        error = exc
    return time.perf_counter() - started, messages, error


def _response_status(messages: list[dict]) -> int | None:
    for message in messages:
        if message["type"] == "http.response.start":
            return message["status"]
    return None


@pytest.mark.asyncio
async def test_timeout_middleware_propagates_errors_before_response() -> None:
    # Regression: unhandled errors before any ASGI send used to wait for the full
    # traffic timeout and then return with no response body at all.
    async def boom(_request):
        raise RuntimeError("boom")

    app = Starlette(
        routes=[Route("/boom", boom)],
        middleware=[Middleware(TimeoutMiddleware, timeout=0.5)],
    )

    elapsed, messages, error = await _asgi_call(app, "/boom")

    assert elapsed < 0.4
    assert _response_status(messages) == 500
    # Starlette may re-raise after emitting the 500 for server-side logging.
    assert error is None or isinstance(error, RuntimeError)


@pytest.mark.asyncio
async def test_timeout_middleware_returns_504_when_request_exceeds_timeout() -> None:
    async def slow(_request):
        await asyncio.sleep(1.0)
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[Route("/slow", slow)],
        middleware=[Middleware(TimeoutMiddleware, timeout=0.1)],
    )

    elapsed, messages, error = await _asgi_call(app, "/slow")

    assert error is None
    assert _response_status(messages) == 504
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_timeout_middleware_allows_successful_responses() -> None:
    async def ok(_request):
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[Route("/ok", ok)],
        middleware=[Middleware(TimeoutMiddleware, timeout=0.5)],
    )

    elapsed, messages, error = await _asgi_call(app, "/ok")

    assert error is None
    assert _response_status(messages) == 200
    assert elapsed < 0.2
