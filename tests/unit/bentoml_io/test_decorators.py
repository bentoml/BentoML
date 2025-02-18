from typing import Generator

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from starlette.testclient import TestClient
from typing_extensions import Annotated

import bentoml
from _bentoml_sdk.validators import TensorSchema
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

    assert (
        numpy_func.input_spec.model_fields["arr"].annotation == npt.NDArray[np.float64]
    )
    assert (
        numpy_func.output_spec.model_fields["root"].annotation == npt.NDArray[np.int64]
    )
    assert (
        TensorSchema(format="numpy-array", dtype="int64")
        in numpy_func.output_spec.model_fields["root"].metadata
    )

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

    assert pandas_func.input_spec.model_fields["df1"].annotation is pd.DataFrame
    assert pandas_func.input_spec.model_fields["df2"].annotation is pd.DataFrame
    assert (
        DataframeSchema(columns=("b",))
        in pandas_func.input_spec.model_fields["df2"].metadata
    )
    assert pandas_func.output_spec.model_fields["root"].annotation is pd.DataFrame
    assert (
        DataframeSchema(orient="columns", columns=("a", "b"))
        in pandas_func.output_spec.model_fields["root"].metadata
    )


def test_api_root_input():
    from _bentoml_sdk.io_models import IORootModel

    @bentoml.api
    def root_input(_, name: str, /) -> str:
        return name

    assert issubclass(root_input.input_spec, IORootModel)
    assert root_input.input_spec.model_fields["root"].annotation is str


def test_api_root_input_illegal():
    with pytest.raises(TypeError):

        @bentoml.api
        def root_input(_, name: str, age: int, /) -> str:
            return name

    with pytest.raises(TypeError):

        @bentoml.api
        def root_input(_, name: str, /, age: int) -> str:
            return name
