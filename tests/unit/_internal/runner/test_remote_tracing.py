from __future__ import annotations

import typing as t

import httpx
import pytest
import pytest_asyncio
from aiohttp import web
from opentelemetry.context import get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import SpanKind
from opentelemetry.util.http import ExcludeList
from simple_di import providers
from starlette.exceptions import HTTPException
from starlette.requests import Request

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.runner.runner_handle.remote import RemoteRunnerClient
from bentoml._internal.server.http_app import HTTPAppFactory


class ProbeRunnable(bentoml.legacy.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    @bentoml.legacy.Runnable.method(batchable=False)
    def predict(self, value: str) -> str:
        return value


@pytest_asyncio.fixture
async def traced_runner(
    monkeypatch: pytest.MonkeyPatch, unused_tcp_port: int
) -> t.AsyncGenerator[
    tuple[
        bentoml.legacy.Runner, RemoteRunnerClient, InMemorySpanExporter, dict[str, int]
    ],
    None,
]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(BentoMLContainer, "tracer_provider", providers.Static(provider))
    monkeypatch.setattr(
        BentoMLContainer,
        "remote_runner_mapping",
        providers.Static({"probe": f"tcp://127.0.0.1:{unused_tcp_port}"}),
    )

    app = web.Application()
    response_status = {"status": 200}

    async def readyz(_: web.Request) -> web.Response:
        return web.Response(status=response_status["status"])

    app.router.add_get("/readyz", readyz)
    app.router.add_get("/predict", readyz)
    server = web.AppRunner(app)
    await server.setup()
    await web.TCPSite(server, "127.0.0.1", unused_tcp_port).start()
    runner = bentoml.legacy.Runner(ProbeRunnable, name="probe")
    runner.init_client()
    client = t.cast(RemoteRunnerClient, runner._runner_handle)
    try:
        yield runner, client, exporter, response_status
    finally:
        if client._client_cache is not None:
            await client._client_cache.close()
        if client._conn is not None:
            await client._conn.close()
            client._conn = None
        await server.cleanup()
        provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "excluded_urls,expected_client_spans",
    [
        ([], 1),
        (["/healthz", "/livez", "/readyz"], 0),
        ([r"/readyz$"], 0),
        ([r"^http://service/readyz$"], 0),
    ],
)
@pytest.mark.parametrize("status", [200, 503])
async def test_readiness_tracing_exclusions(
    traced_runner: tuple[
        bentoml.legacy.Runner, RemoteRunnerClient, InMemorySpanExporter, dict[str, int]
    ],
    monkeypatch: pytest.MonkeyPatch,
    excluded_urls: list[str],
    expected_client_spans: int,
    status: int,
) -> None:
    runner, client, exporter, runner_app = traced_runner
    runner_app["status"] = status
    monkeypatch.setattr(
        BentoMLContainer,
        "tracing_excluded_urls",
        providers.Static(ExcludeList(excluded_urls)),
    )
    service = bentoml.legacy.Service("probe_service", runners=[runner])
    app = HTTPAppFactory(service, enable_metrics=False)()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as api_client:
        response = await api_client.get("/readyz")
    assert response.status_code == status
    client_spans = [
        span for span in exporter.get_finished_spans() if span.kind == SpanKind.CLIENT
    ]
    assert len(client_spans) == expected_client_spans

    # Suppressing a probe must not suppress subsequent inference requests.
    exporter.clear()
    async with client._client.get(f"{client._addr}/predict") as response:
        await response.read()
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503])
async def test_readiness_restores_tracing_context(
    traced_runner: tuple[
        bentoml.legacy.Runner, RemoteRunnerClient, InMemorySpanExporter, dict[str, int]
    ],
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    runner, client, exporter, runner_app = traced_runner
    runner_app["status"] = status
    monkeypatch.setattr(
        BentoMLContainer,
        "tracing_excluded_urls",
        providers.Static(ExcludeList(["/readyz"])),
    )
    factory = HTTPAppFactory(
        bentoml.legacy.Service("probe_service", runners=[runner]), enable_metrics=False
    )
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("service", 80),
            "path": "/readyz",
            "query_string": b"",
            "headers": [],
        }
    )
    original_context = get_current()
    if status == 503:
        with pytest.raises(HTTPException) as exc_info:
            await factory.readyz(request)
        assert exc_info.value.status_code == 503
    else:
        assert (await factory.readyz(request)).status_code == 200
    assert get_current() is original_context

    exporter.clear()
    async with client._client.get(f"{client._addr}/predict") as response:
        await response.read()
    assert len(exporter.get_finished_spans()) == 1
