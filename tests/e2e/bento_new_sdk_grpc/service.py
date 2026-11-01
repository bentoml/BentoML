from __future__ import annotations

import typing as t

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

import bentoml
from bentoml.validators import TensorSchema

Array = t.Annotated[npt.NDArray[np.float32], TensorSchema("numpy-array")]


class ServiceArgs(BaseModel):
    greeting: str


args = bentoml.use_arguments(ServiceArgs)


@bentoml.service
class MyService:
    @bentoml.api
    def greet(self, name: str) -> str:
        return f"{args.greeting} {name}"

    @bentoml.api
    def predict(self, x: Array) -> Array:
        return x * 2

    @bentoml.api
    def context_greet(self, name: str, ctx: bentoml.Context) -> str:
        ctx.response.headers["x-response-source"] = "bentoml-context"
        return f"{ctx.request.headers['x-request-source']} {name}"

    @bentoml.api
    async def stream_greet(self, name: str) -> t.AsyncGenerator[str, None]:
        yield f"hello {name}"
