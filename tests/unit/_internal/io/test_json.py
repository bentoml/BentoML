import json
import typing as t
from dataclasses import dataclass

import numpy as np
import pydantic
import pytest


@dataclass
class _ExampleSchema:
    name: str
    endpoints: t.List[str]


class _Schema(pydantic.BaseModel):
    name: str
    endpoints: t.List[str]


@pytest.mark.parametrize(
    "obj",
    [
        _ExampleSchema(name="test", endpoints=["predict", "health"]),
        _Schema(name="test", endpoints=["predict", "health"]),
        np.array([[1]]),
    ],
)
def test_json_encoder(obj: t.Any) -> None:
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
        dumped == '{"name":"test","endpoints":["predict","health"]}'
        or dumped == "[[1]]"
    )
