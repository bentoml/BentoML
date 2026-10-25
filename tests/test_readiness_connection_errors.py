"""
Tests: is_ready() must report not-ready (False) instead of raising when the
runner cannot be reached.

`is_ready` feeds readiness probes (runner_probe / /readyz aggregates its
results with asyncio.gather), so a runner that is simply not listening yet
must read as False, not crash the probe with httpx.ConnectError or
aiohttp.ClientConnectorError. `wait_until_server_ready` in the same module
already treats (TimeoutException, ConnectError) as retryable, and
server/proxy.py's health check catches aiohttp.ClientError -> False; the
is_ready implementations only caught timeout variants, so a refused
connection propagated out of the probe.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import httpx

    from _bentoml_impl.client.http import AsyncHTTPClient
    from _bentoml_impl.client.http import SyncHTTPClient
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class TestSyncHTTPClientConnectError:
    def test_is_ready_returns_false_on_connect_error(self):
        # Bypass __init__: the constructor probes the server (wait_until_server_ready)
        # and this test is about is_ready alone.
        client = SyncHTTPClient.__new__(SyncHTTPClient)
        client._readyz_endpoint = "/readyz"
        client.client = MagicMock()
        client.client.get.side_effect = httpx.ConnectError("connection refused")

        assert client.is_ready(timeout=1) is False


class TestAsyncHTTPClientConnectError:
    @pytest.mark.asyncio
    async def test_is_ready_returns_false_on_connect_error(self):
        client = AsyncHTTPClient(url="http://test.example.com")
        client._readyz_endpoint = "/readyz"
        client.client = AsyncMock()
        client.client.get.side_effect = httpx.ConnectError("connection refused")

        assert await client.is_ready(timeout=1) is False
        client.client.get.assert_called_once()


class TestProxyAndRemoteConnectErrors:
    @pytest.mark.asyncio
    async def test_proxy2_is_ready_returns_false_on_client_error(self):
        aiohttp = pytest.importorskip("aiohttp")

        from _bentoml_impl.client.proxy2 import AsyncClient as ProxyAsyncClient

        client = ProxyAsyncClient.__new__(ProxyAsyncClient)
        client._readyz_endpoint = "/readyz"
        client._session_manager = MagicMock()
        session = AsyncMock()
        session.get.side_effect = aiohttp.ClientConnectorError(
            connection_key=MagicMock(), os_error=OSError("connection refused")
        )
        client._session_manager.get_session = AsyncMock(return_value=session)

        assert await client.is_ready(timeout=1) is False

    @pytest.mark.asyncio
    async def test_remote_handle_is_ready_returns_false_on_client_error(self):
        aiohttp = pytest.importorskip("aiohttp")

        from bentoml._internal.runner.runner_handle import remote as remote_mod

        handle = remote_mod.RemoteRunnerClient.__new__(remote_mod.RemoteRunnerClient)
        handle._conn = None  # keep __del__ quiet: no real connection was opened
        handle._runner = MagicMock()
        handle._addr = "http://test.example.com"
        # _client is a property that builds a real session unless a cached one is
        # alive on the running loop; hand it our failing mock instead.
        session = AsyncMock()
        session.closed = False
        # A refused connection surfaces when the request is issued, before any
        # response context manager is entered, so a plain MagicMock suffices.
        session.get = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("connection refused")
            )
        )
        handle._loop = asyncio.get_running_loop()
        handle._client_cache = session

        assert await handle.is_ready(timeout=1) is False
