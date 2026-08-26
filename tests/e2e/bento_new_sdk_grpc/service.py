from __future__ import annotations

import typing as t

import numpy as np
import numpy.typing as npt

import bentoml
from bentoml.validators import TensorSchema

Array = t.Annotated[npt.NDArray[np.float32], TensorSchema("numpy-array")]


@bentoml.service
class MyService:
    @bentoml.api
    def greet(self, name: str) -> str:
        return f"hello {name}"

    @bentoml.api
    def predict(self, x: Array) -> Array:
        return x * 2

    @bentoml.api
    async def stream_greet(self, name: str) -> t.AsyncGenerator[str, None]:
        yield f"hello {name}"
