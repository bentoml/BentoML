from typing import Generator

import pytest
from starlette.testclient import TestClient

import bentoml


def test_mount_asgi_app():
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.mount_asgi_app(app, path="/test")
    @bentoml.service(metrics={"enabled": False})
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    with TestClient(app=TestService.to_asgi()) as client:
        response = client.get("/test/hello")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, world!"


def test_mount_asgi_app_later():
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.service(metrics={"enabled": False})
    @bentoml.mount_asgi_app(app, path="/test")
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    with TestClient(app=TestService.to_asgi()) as client:
        response = client.get("/test/hello")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, world!"


def test_service_instantiate():
    @bentoml.service
    class TestService:
        @bentoml.api
        def hello(self, name: str) -> str:
            return f"Hello, {name}!"

        @bentoml.api
        def stream(self, name: str) -> Generator[str, None, None]:
            for i in range(3):
                yield f"Hello, {name}! {i}"

    svc = TestService()
    assert svc.hello("world") == "Hello, world!"
    assert list(svc.stream("world")) == [
        "Hello, world! 0",
        "Hello, world! 1",
        "Hello, world! 2",
    ]


@pytest.mark.asyncio
async def test_service_instantiate_to_async():
    @bentoml.service
    class TestService:
        @bentoml.api
        def hello(self, name: str) -> str:
            return f"Hello, {name}!"

        @bentoml.api
        def stream(self, name: str) -> Generator[str, None, None]:
            for i in range(3):
                yield f"Hello, {name}! {i}"

    svc = TestService()
    assert await svc.to_async.hello("world") == "Hello, world!"
    assert [text async for text in svc.to_async.stream("world")] == [
        "Hello, world! 0",
        "Hello, world! 1",
        "Hello, world! 2",
    ]
