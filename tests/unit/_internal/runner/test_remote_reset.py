from __future__ import annotations

import typing as t

import pytest
import pytest_asyncio
from aiohttp import web
from simple_di import providers

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.runner.runner_handle.remote import RemoteRunnerClient


class ResetRunnable(bentoml.legacy.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    @bentoml.legacy.Runnable.method(batchable=False)
    def predict(self, value: str) -> str:
        return value


@pytest_asyncio.fixture
async def remote_client(
    monkeypatch: pytest.MonkeyPatch, unused_tcp_port: int
) -> t.AsyncGenerator[tuple[RemoteRunnerClient, list[object]], None]:
    connections: list[object] = []

    async def readyz(request: web.Request) -> web.Response:
        assert request.transport is not None
        connections.append(request.transport)
        return web.Response(text="ready")

    app = web.Application()
    app.router.add_get("/readyz", readyz)
    server = web.AppRunner(app)
    await server.setup()
    await web.TCPSite(server, "127.0.0.1", unused_tcp_port).start()
    monkeypatch.setattr(
        BentoMLContainer,
        "remote_runner_mapping",
        providers.Static({"reset_probe": f"tcp://127.0.0.1:{unused_tcp_port}"}),
    )
    runner = bentoml.legacy.Runner(ResetRunnable, name="reset_probe")
    client = RemoteRunnerClient(runner)
    try:
        yield client, connections
    finally:
        if client._client_cache is not None:
            await client._client_cache.close()
        if client._conn is not None:
            await client._conn.close()
            client._conn = None
        await server.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("warm_connection", [False, True])
async def test_reset_closes_connector_and_reconnects(
    remote_client: tuple[RemoteRunnerClient, list[object]], warm_connection: bool
) -> None:
    client, connections = remote_client
    session = client._client
    connector = session.connector
    assert connector is not None
    if warm_connection:
        assert await client.is_ready(timeout=1)

    await client._reset_client()

    assert session.closed
    assert connector.closed
    # Resetting an already-reset client must also be safe.
    await client._reset_client()
    assert await client.is_ready(timeout=1)
    assert client._client is not session
    assert client._client.connector is not connector
    if warm_connection:
        assert connections[-1] is not connections[0]


@pytest.mark.asyncio
async def test_reset_before_first_request(
    remote_client: tuple[RemoteRunnerClient, list[object]],
) -> None:
    client, connections = remote_client
    await client._reset_client()
    await client._reset_client()
    assert await client.is_ready(timeout=1)
    assert len(connections) == 1
