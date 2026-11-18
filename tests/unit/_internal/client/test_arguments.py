from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from _bentoml_impl.client.http import AsyncHTTPClient
from _bentoml_impl.client.http import SyncHTTPClient
from _bentoml_impl.client.proxy2 import AsyncClient


@pytest.fixture
def app() -> Starlette:
    async def schema(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "routes": [
                    {
                        "name": "predict",
                        "route": "/predict",
                        "input": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        "output": {"type": "string"},
                        "is_task": True,
                    },
                    {
                        "name": "empty",
                        "route": "/empty",
                        "input": {"type": "object", "properties": {}},
                        "output": {"type": "string"},
                        "is_task": True,
                    },
                    {
                        "name": "root",
                        "route": "/root",
                        "input": {
                            "type": "object",
                            "root_input": True,
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        "output": {"type": "string"},
                    },
                ]
            }
        )

    async def predict(request: Request) -> PlainTextResponse | JSONResponse:
        app.state.requests.append((request.url.path, await request.json()))
        if request.url.path.endswith("/submit"):
            return JSONResponse({"task_id": "test-task"})
        return PlainTextResponse("prediction")

    app = Starlette(
        routes=[
            Route("/schema.json", schema),
            Route("/predict", predict, methods=["POST"]),
            Route("/predict/submit", predict, methods=["POST"]),
            Route("/empty", predict, methods=["POST"]),
            Route("/empty/submit", predict, methods=["POST"]),
            Route("/root", predict, methods=["POST"]),
        ]
    )
    app.state.requests = []
    return app


@pytest.mark.parametrize(
    "endpoint,args", [("predict", ("kept", "lost")), ("empty", ("lost",))]
)
def test_sync_client_rejects_extra_positional_arguments(
    app: Starlette, endpoint: str, args: tuple[str, ...]
) -> None:
    with SyncHTTPClient("http://localhost", app=app, token="") as client:
        with pytest.raises(TypeError, match="Too many positional arguments"):
            client.call(endpoint, *args)
    assert app.state.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("client_cls", [AsyncHTTPClient, AsyncClient])
@pytest.mark.parametrize(
    "endpoint,args", [("predict", ("kept", "lost")), ("empty", ("lost",))]
)
@pytest.mark.parametrize("submit", [False, True])
async def test_async_client_rejects_extra_positional_arguments(
    app: Starlette,
    client_cls: type[AsyncHTTPClient] | type[AsyncClient],
    endpoint: str,
    args: tuple[str, ...],
    submit: bool,
) -> None:
    async with client_cls("http://localhost", app=app, token="") as client:
        with pytest.raises(TypeError, match="Too many positional arguments"):
            if submit:
                await getattr(client, endpoint).submit(*args)
            else:
                await client.call(endpoint, *args)
    assert app.state.requests == []


def test_sync_client_accepts_valid_arguments(app: Starlette) -> None:
    with SyncHTTPClient("http://localhost", app=app, token="") as client:
        assert client.predict("hello") == "prediction"
        assert client.predict(text="world") == "prediction"
        assert client.empty() == "prediction"
        assert client.root({"text": "root"}) == "prediction"
    assert app.state.requests == [
        ("/predict", {"text": "hello"}),
        ("/predict", {"text": "world"}),
        ("/empty", {}),
        ("/root", {"text": "root"}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_cls", [AsyncHTTPClient, AsyncClient])
async def test_async_client_accepts_valid_arguments(
    app: Starlette, client_cls: type[AsyncHTTPClient] | type[AsyncClient]
) -> None:
    async with client_cls("http://localhost", app=app, token="") as client:
        assert await client.predict("hello") == "prediction"
        assert await client.predict(text="world") == "prediction"
        assert await client.empty() == "prediction"
        assert await client.root({"text": "root"}) == "prediction"
        if client_cls is AsyncHTTPClient:
            assert (await client.predict.submit("task")).id == "test-task"
    assert app.state.requests == [
        ("/predict", {"text": "hello"}),
        ("/predict", {"text": "world"}),
        ("/empty", {}),
        ("/root", {"text": "root"}),
    ] + (
        [("/predict/submit", {"text": "task"})] if client_cls is AsyncHTTPClient else []
    )
