from typing import Generator

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from starlette.testclient import TestClient
from typing_extensions import Annotated

import bentoml
from bentoml.validators import DataframeSchema
from bentoml.validators import DType
from bentoml.validators import Shape


def test_mount_asgi_app():
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.asgi_app(app, path="/test")
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
    @bentoml.asgi_app(app, path="/test")
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


def test_api_decorator_numpy():
    @bentoml.api
    def numpy_func(
        _,  # The decorator assumes `self` is the first arg.
        arr: npt.NDArray[np.float64],
    ) -> Annotated[npt.NDArray[np.int64], DType("int64"), Shape((1,))]:
        return arr.astype(np.int64)

    numpy_func.input_spec.model_fields["arr"].annotation is npt.NDArray[np.float64]
    numpy_func.output_spec.model_fields["root"].annotation is Annotated[
        npt.NDArray[np.int64], DType("int64"), Shape((1,))
    ]

    with pytest.raises(
        TypeError,
        match=r"Unable to infer the output spec for function .+, please specify output_spec manually",
    ):

        @bentoml.api
        def numpy_func(
            _,  # The decorator assumes `self` is the first arg.
            arr: npt.NDArray[np.float64],
        ) -> Annotated[npt.NDArray[np.float64], DType("int64"), Shape((1,))]:
            return arr.astype(np.int64)


def test_api_decorator_pandas():
    @bentoml.api
    def pandas_func(
        _,  # The decorator assumes `self` is the first arg.
        df1: pd.DataFrame,
        df2: Annotated[pd.DataFrame, DataframeSchema(columns=("b",))],
    ) -> Annotated[
        pd.DataFrame,
        DataframeSchema(orient="columns", columns=["a", "b"]),
    ]:
        return pd.concat([df1, df2], axis=1)

    pandas_func.input_spec.model_fields["df1"].annotation is pd.DataFrame
    pandas_func.input_spec.model_fields["df2"].annotation is Annotated[
        pd.DataFrame,
        DataframeSchema(columns=("b",)),
    ]
    pandas_func.output_spec.model_fields["root"].annotation is Annotated[
        pd.DataFrame,
        DataframeSchema(orient="columns", columns=("a", "b")),
    ]
