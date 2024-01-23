import inspect
from typing import Generator

import pytest

import bentoml
from _bentoml_sdk.io_models import IODescriptor


@bentoml.api.sync_to_async
def add(a: int, b: int) -> int:
    return a + b


@bentoml.api.sync_to_async(threads=1)
def multiply(a: int, b: int) -> int:
    return a * b


@bentoml.api.sync_to_async
def my_generator(end: int) -> Generator[int, None, None]:
    for i in range(end):
        yield i


@pytest.mark.asyncio
async def test_sync_to_async():
    assert inspect.iscoroutinefunction(add)
    assert inspect.iscoroutinefunction(multiply)
    assert inspect.isasyncgenfunction(my_generator)

    assert await add(1, 2) == 3
    assert await multiply(2, 3) == 6
    assert [i async for i in my_generator(3)] == [0, 1, 2]

    imodel = IODescriptor.from_input(add)
    omodel = IODescriptor.from_output(add)
    assert set(imodel.model_fields) == {"a", "b"}
    assert omodel(4).root == 4
