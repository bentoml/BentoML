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
    "obj",
    [
        _ExampleSchema(name="test", endpoints=["predict", "health"]),
        _Schema(name="test", endpoints=["predict", "health"]),
        test_arr,
    ],
)
def test_json_encoder(
    obj: t.Union[_ExampleSchema, pydantic.BaseModel, "np.ndarray[t.Any, np.dtype[t.Any]]"]
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
    assert (
        dumped == '{"name":"test","endpoints":["predict","health"]}' or dumped == "[[1]]"
    )
