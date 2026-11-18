from __future__ import annotations

import asyncio
import typing as t
from types import SimpleNamespace

import aiohttp
import httpx
import pytest
import pytest_asyncio
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.routing import Route

from _bentoml_impl.client import http
from _bentoml_impl.client import proxy2
from bentoml.exceptions import ServiceUnavailable


class ReadinessProbe:
    def __init__(self) -> None:
        self.now = 0.0
        self.request_duration = 0.03125
        self.requests: list[float] = []
        self.pauses: list[float] = []
        self.ready_after: int | None = None
        self.app = Starlette(
            routes=[Route("/readyz", self.ready), Route("/schema.json", self.schema)]
        )

    async def ready(self, request: t.Any) -> Response:
        self.requests.append(self.now)
        self.now += self.request_duration
        ready = self.ready_after is not None and len(self.requests) >= self.ready_after
        return Response(status_code=200 if ready else 503)

    async def schema(self, request: t.Any) -> Response:
        return JSONResponse({"routes": []})

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        assert delay > 0
        self.pauses.append(delay)
        self.now += delay

    async def async_sleep(self, delay: float) -> None:
        self.sleep(delay)
        await asyncio.sleep(0)


Client = t.Union[http.SyncHTTPClient, http.AsyncHTTPClient, proxy2.AsyncClient]


async def wait(client: Client, timeout: float) -> None:
    if isinstance(client, http.SyncHTTPClient):
        client.wait_until_server_ready(timeout)
    else:
        await client.wait_until_server_ready(timeout)


@pytest_asyncio.fixture(params=["sync-httpx", "async-httpx", "aiohttp"])
async def readiness_client(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> t.AsyncGenerator[tuple[Client, ReadinessProbe], None]:
    probe = ReadinessProbe()
    client_type = {
        "sync-httpx": http.SyncHTTPClient,
        "async-httpx": http.AsyncHTTPClient,
        "aiohttp": proxy2.AsyncClient,
    }[request.param]
    client = client_type("http://localhost", app=probe.app)
    module = proxy2 if request.param == "aiohttp" else http
    # Control only the readiness clock and pauses, leaving HTTP transports and
    # the event loop's clock intact.
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=probe.monotonic, sleep=probe.sleep)
    )
    monkeypatch.setattr(
        module, "asyncio", SimpleNamespace(sleep=probe.async_sleep), raising=False
    )
    try:
        yield client, probe
    finally:
        if isinstance(client, http.SyncHTTPClient):
            client.close()
        else:
            await client.close()


@pytest.mark.asyncio
async def test_readiness_polls_are_paced(
    readiness_client: tuple[Client, ReadinessProbe],
) -> None:
    client, probe = readiness_client
    with pytest.raises(
        ServiceUnavailable, match="Server is not ready after 0.25 seconds"
    ):
        await wait(client, 0.25)
    assert 1 < len(probe.requests) <= 3
    assert probe.requests[0] == 0
    assert probe.pauses
    assert all(0 < delay <= 0.1 for delay in probe.pauses)
    assert probe.now == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_readiness_recovers_after_unsuccessful_polls(
    readiness_client: tuple[Client, ReadinessProbe],
) -> None:
    client, probe = readiness_client
    probe.ready_after = 3
    await wait(client, 1)
    assert len(probe.requests) == 3
    assert len(probe.pauses) == 2
    assert all(b - a >= 0.1 for a, b in zip(probe.requests, probe.requests[1:]))


@pytest.mark.asyncio
async def test_ready_server_has_no_added_pause(
    readiness_client: tuple[Client, ReadinessProbe],
) -> None:
    client, probe = readiness_client
    probe.ready_after = 1
    await wait(client, 1)
    assert probe.requests == [0]
    assert probe.pauses == []


@pytest.mark.asyncio
async def test_readiness_pause_respects_remaining_time(
    readiness_client: tuple[Client, ReadinessProbe],
) -> None:
    client, probe = readiness_client
    with pytest.raises(ServiceUnavailable):
        await wait(client, 0.0625)
    assert probe.requests == [0]
    assert probe.pauses == pytest.approx([0.03125])
    assert probe.now == pytest.approx(0.0625)


@pytest.mark.asyncio
async def test_no_pause_after_request_exhausts_deadline(
    readiness_client: tuple[Client, ReadinessProbe],
) -> None:
    client, probe = readiness_client
    probe.request_duration = 0.4
    with pytest.raises(ServiceUnavailable):
        await wait(client, 0.25)
    assert probe.requests == [0]
    assert probe.pauses == []


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1])
async def test_nonpositive_readiness_timeout_does_not_poll(
    readiness_client: tuple[Client, ReadinessProbe], timeout: float
) -> None:
    client, probe = readiness_client
    with pytest.raises(ServiceUnavailable):
        await wait(client, timeout)
    assert probe.requests == []
    assert probe.pauses == []


@pytest.mark.asyncio
async def test_connection_errors_are_paced(
    readiness_client: tuple[Client, ReadinessProbe], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, probe = readiness_client

    def fail_request(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:
        probe.requests.append(probe.now)
        probe.now += probe.request_duration
        if isinstance(client, proxy2.AsyncClient):
            raise aiohttp.ClientConnectionError("not listening yet")
        raise httpx.ConnectError("not listening yet")

    async def async_fail_request(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:
        fail_request(*args, **kwargs)

    if isinstance(client, proxy2.AsyncClient):
        session = await client._session_manager.get_session()
        monkeypatch.setattr(session, "get", fail_request)
    elif isinstance(client, http.SyncHTTPClient):
        monkeypatch.setattr(client.client, "get", fail_request)
    else:
        monkeypatch.setattr(client.client, "get", async_fail_request)

    with pytest.raises(ServiceUnavailable):
        await wait(client, 0.25)
    assert 1 < len(probe.requests) <= 3
    assert probe.pauses
    assert probe.now == pytest.approx(0.25)


@pytest.mark.asyncio
@pytest.mark.parametrize("readiness_client", ["async-httpx", "aiohttp"], indirect=True)
async def test_async_readiness_can_be_cancelled_during_pause(
    readiness_client: tuple[Client, ReadinessProbe], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, probe = readiness_client
    paused = asyncio.Event()

    async def pause(delay: float) -> None:
        paused.set()
        await asyncio.Future()

    module = proxy2 if isinstance(client, proxy2.AsyncClient) else http
    monkeypatch.setattr(module.asyncio, "sleep", pause)
    task = asyncio.create_task(wait(client, 1))
    try:
        await asyncio.wait_for(paused.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert probe.requests == [0]
