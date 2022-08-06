import json
import typing as t
from dataclasses import dataclass

import numpy as np
import pytest
import pydantic


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
        (None, None),
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
