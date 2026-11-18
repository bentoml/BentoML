import aiohttp
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from _bentoml_impl.client.proxy2 import SessionManager


@pytest.mark.asyncio
async def test_session_refresh_creates_new_connector() -> None:
    # Regression test: fresh connector created on session refresh.
    # Before fix: closing old session closed shared connector, causing
    # "Session is closed" on subsequent requests.
    app = Starlette(routes=[Route("/", lambda r: JSONResponse({"ok": True}))])
    manager = SessionManager(
        url="http://127.0.0.1:3000",
        timeout=30.0,
        headers={},
        app=app,
        max_requests=1,
    )
    try:
        session1 = await manager.get_session()
        async with session1.get("/") as resp:
            assert resp.status == 200
        session2 = await manager.get_session()
        async with session2.get("/") as resp:
            assert resp.status == 200
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_tcp_connector_has_explicit_keepalive_timeout() -> None:
    # Regression test for #4971: the tcp-scheme branch of _make_client
    # previously left connector=None, so aiohttp fell back to its default
    # keepalive_timeout (15s), which could exceed the server's uvicorn
    # timeout_keep_alive default (5s) and cause intermittent
    # ServerDisconnectedError/RemoteProtocolError under load.
    manager = SessionManager(
        url="tcp://127.0.0.1:3000",
        timeout=30.0,
        headers={},
    )
    try:
        session = await manager.get_session()
        connector = session.connector
        assert isinstance(connector, aiohttp.TCPConnector)
        assert connector._keepalive_timeout == 4.0
    finally:
        await manager.close()
