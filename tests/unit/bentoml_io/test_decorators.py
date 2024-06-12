from typing import Annotated
from typing import Any
from typing import Generator

import numpy as np
import numpy.typing as npt
import pytest

import bentoml
from bentoml.validators import DType
from bentoml.validators import Shape


@pytest.mark.asyncio
async def test_mount_asgi_app():
    import httpx
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.mount_asgi_app(app, path="/test")
    @bentoml.service(metrics={"enabled": False})
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    async with httpx.AsyncClient(
        app=TestService.to_asgi(init=True), base_url="http://testserver"
    ) as client:
        response = await client.get("/test/hello")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, world!"


@pytest.mark.asyncio
async def test_mount_asgi_app_later():
    import httpx
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.service(metrics={"enabled": False})
    @bentoml.mount_asgi_app(app, path="/test")
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    async with httpx.AsyncClient(
        app=TestService.to_asgi(init=True), base_url="http://testserver"
    ) as client:
        response = await client.get("/test/hello")
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

        @bentoml.api
        def numpy(
            self,
            arr: npt.NDArray[np.float64],
        ) -> Annotated[npt.NDArray[Any], DType("float64"), Shape((1,))]:
            return arr

    svc = TestService()
    assert svc.hello("world") == "Hello, world!"
    assert list(svc.stream("world")) == [
        "Hello, world! 0",
        "Hello, world! 1",
        "Hello, world! 2",
    ]
    np.testing.assert_array_equal(
        svc.numpy(np.array([1.0, 2.0, 3.0])),
        np.array([1.0, 2.0, 3.0]),
    )


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

        @bentoml.api
        def numpy(
            self,
            arr: npt.NDArray[np.float64],
        ) -> Annotated[npt.NDArray[Any], DType("float64"), Shape((1,))]:
            return arr

    svc = TestService()
    assert await svc.to_async.hello("world") == "Hello, world!"
    assert [text async for text in svc.to_async.stream("world")] == [
        "Hello, world! 0",
        "Hello, world! 1",
        "Hello, world! 2",
    ]
    np.testing.assert_array_equal(
        await svc.to_async.numpy(np.array([1.0, 2.0, 3.0])),
        np.array([1.0, 2.0, 3.0]),
    )
