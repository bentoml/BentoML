import json
import typing as t
from dataclasses import dataclass
import asyncio

import numpy as np
from bentoml._internal.io_descriptors.json import JSON
import pytest
import pydantic


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@dataclass
class _ExampleSchema:
    name: str
    endpoints: t.List[str]


class _Schema(pydantic.BaseModel):
    name: str
    endpoints: t.List[str]


test_arr = t.cast("np.ndarray[t.Any, np.dtype[np.int32]]", np.array([[1]]))


@pytest.mark.parametrize(
    "obj,expected",
    [
        (
            _ExampleSchema(name="test", endpoints=["predict", "health"]),
            '{"name":"test","endpoints":["predict","health"]}',
        ),
        (
            _Schema(name="test", endpoints=["predict", "health"]),
            '{"name":"test","endpoints":["predict","health"]}',
        ),
        (test_arr, "[[1]]"),
    ],
)
def test_json_encoder(
    obj: t.Union[
        _ExampleSchema, pydantic.BaseModel, "np.ndarray[t.Any, np.dtype[t.Any]]", None
    ],
    expected: t.Union[str, None],
) -> None:
    from bentoml._internal.io_descriptors.json import DefaultJsonEncoder

    dumped = json.dumps(
        obj,
        cls=DefaultJsonEncoder,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    )
    assert expected == dumped


def test_json_description_to_http_response(loop: asyncio.AbstractEventLoop):

    json_description = JSON()

    response = loop.run_until_complete(json_description.to_http_response(None))

    assert b"" == response.body
