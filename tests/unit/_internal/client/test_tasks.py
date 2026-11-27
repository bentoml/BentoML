import asyncio
import json

import anyio
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.routing import Route

import bentoml
from _bentoml_impl.client.proxy2 import AsyncClient
from _bentoml_impl.client.proxy2 import RemoteProxy
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException


@bentoml.service
class TaskService:
    @bentoml.task
    async def predict(self, value: str) -> str:
        return value


@pytest.fixture(autouse=True)
def services_config():
    original = BentoMLContainer.config.get()
    BentoMLContainer.config.set({**original, "services": {}})
    try:
        yield
    finally:
        BentoMLContainer.config.set(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_kind", ["async", "proxy_async", "proxy_sync"])
async def test_submit_returns_task(client_kind: str) -> None:
    submissions = []

    async def submit(request):
        submissions.append(await request.json())
        return JSONResponse({"task_id": "submitted-task"}, status_code=202)

    app = Starlette(routes=[Route("/predict/submit", submit, methods=["POST"])])
    if client_kind == "async":
        client = AsyncClient("http://localhost:3000", service=TaskService, app=app)
    else:
        client = RemoteProxy(
            "http://localhost:3000",
            service=TaskService,
            media_type="application/json",
            app=app,
        )

    await client.__aenter__()
    try:
        if client_kind == "proxy_sync":
            task = await anyio.to_thread.run_sync(
                lambda: client.to_sync.predict.submit(value="hello")
            )
        else:
            task = await client.predict.submit(value="hello")
        assert task.id == "submitted-task"
        assert submissions == [{"value": "hello"}]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body,error",
    [
        (503, "unavailable", BentoMLException),
        (202, "not json", json.JSONDecodeError),
        (202, "{}", KeyError),
    ],
)
async def test_submit_response_errors(status, body, error) -> None:
    submissions = []

    async def submit(request):
        submissions.append(await request.json())
        return Response(body, status_code=status, media_type="application/json")

    app = Starlette(routes=[Route("/predict/submit", submit, methods=["POST"])])
    async with AsyncClient(
        "http://localhost:3000", service=TaskService, app=app
    ) as client:
        with pytest.raises(error) as exc:
            await client.predict.submit(value="hello")
        if status == 503:
            assert exc.value.error_code == 503
            assert "unavailable" in str(exc.value)
        assert submissions == [{"value": "hello"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_submit_waits_for_response_body(cancel: bool) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    submissions = []

    async def submit(request):
        submissions.append(await request.json())
        response = web.StreamResponse(
            status=202, headers={"Content-Type": "application/json"}
        )
        await response.prepare(request)
        await response.write(b'{"task_id":')
        started.set()
        await release.wait()
        if not cancel:
            await response.write(b'"streamed-task"}')
        return response

    app = web.Application()
    app.router.add_post("/predict/submit", submit)
    async with TestServer(app) as server:
        async with AsyncClient(
            str(server.make_url("")), service=TaskService, server_ready_timeout=0
        ) as client:
            submission = asyncio.create_task(client.predict.submit(value="hello"))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                if cancel:
                    submission.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await submission
                else:
                    release.set()
                    task = await asyncio.wait_for(submission, timeout=5)
                    assert task.id == "streamed-task"
                assert submissions == [{"value": "hello"}]
            finally:
                release.set()
                if not submission.done():
                    submission.cancel()
                await asyncio.gather(submission, return_exceptions=True)
