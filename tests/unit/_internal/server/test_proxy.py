from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import mock

import aiohttp
import pytest
from starlette.requests import Request

from _bentoml_impl.server import proxy


class _ResponseContent:
    async def iter_any(self):
        if False:
            yield b""


class _UpstreamResponse:
    status = 200
    headers: dict[str, str] = {}
    content = _ResponseContent()


@contextlib.asynccontextmanager
async def _upstream_response():
    yield _UpstreamResponse()


@pytest.mark.asyncio
async def test_remote_proxy_uses_configured_host_and_supplied_client(
    monkeypatch,
) -> None:
    service = mock.MagicMock()
    service.config = {
        "http": {"proxy_host": "upstream.internal", "proxy_port": 9100},
        "endpoints": {"livez": "/ready"},
    }
    service.has_custom_command.return_value = True
    service.context.state = {}

    supplied_client = mock.Mock(spec=aiohttp.ClientSession)
    supplied_client.request.return_value = _upstream_response()
    server_instance = SimpleNamespace(client=supplied_client)
    monkeypatch.setattr(proxy, "get_current_service", lambda: server_instance)

    health_check = mock.AsyncMock(return_value=True)
    monkeypatch.setattr(proxy, "_check_health", health_check)
    open_process = mock.AsyncMock()
    monkeypatch.setattr(proxy.anyio, "open_process", open_process)

    app = proxy.create_proxy_app(service)
    async with app.router.lifespan_context(app):
        assert app.state.client is supplied_client
        assert app.state.proxy_url == "http://upstream.internal:9100"
        health_check.assert_awaited_once_with(
            supplied_client, "http://upstream.internal:9100/ready"
        )
        open_process.assert_not_awaited()

        route = next(route for route in app.routes if route.path == "/{path:path}")
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/claims/42",
            "raw_path": b"/claims/42",
            "query_string": b"detail=full",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request(scope, receive)
        response = await route.endpoint(request, "claims/42")

        requested_url = supplied_client.request.call_args.kwargs["url"]
        assert str(requested_url) == (
            "http://upstream.internal:9100/claims/42?detail=full"
        )
        assert response.status_code == 200
        await response.background()
