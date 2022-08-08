from __future__ import annotations

import json
import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING
from functools import partial
from dataclasses import dataclass

import attr
import numpy as np
import pandas as pd
import pytest
import pydantic

from bentoml.io import JSON
from bentoml._internal.io_descriptors.json import DefaultJsonEncoder

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from bentoml._internal.service.openapi.specification import Schema


@dataclass
class ExampleDataclass:
    name: str
    endpoints: t.List[str]


class ExampleGeneric(str, np.generic):
    pass


@attr.define
class ExampleAttrsClass:
    name: str
    endpoints: t.List[str]


class BaseSchema(pydantic.BaseModel):
    name: str
    endpoints: t.List[str]


def test_json_description_to_http_response():
    loop = asyncio.new_event_loop()
    try:
        response = loop.run_until_complete(JSON().to_http_response(None))
        assert b"" == response.body
    finally:
        loop.close()


class Nested(pydantic.BaseModel):
    toplevel: str
    nested: BaseSchema

    class Config:
        extra = "allow"


dumps = partial(
    json.dumps,
    cls=DefaultJsonEncoder,
    ensure_ascii=False,
    indent=None,
    separators=(",", ":"),
)


def test_json_encoder_dataclass_like():
    expected = '{"name":"test","endpoints":["predict","health"]}'
    assert (
        dumps(ExampleDataclass(name="test", endpoints=["predict", "health"]))
        == expected
    )
    assert dumps(BaseSchema(name="test", endpoints=["predict", "health"])) == expected
    assert (
        dumps(ExampleAttrsClass(name="test", endpoints=["predict", "health"]))
        == expected
    )

    assert (
        dumps(
            Nested(
                toplevel="test",
                nested=BaseSchema(name="test", endpoints=["predict", "health"]),
            )
        )
        == '{"toplevel":"test","nested":{"name":"test","endpoints":["predict","health"]}}'
    )


def test_json_encoder_numpy():
    assert dumps(np.array([[1]])) == "[[1]]"
    assert dumps(ExampleGeneric("asdf")) == '"asdf"'


def test_json_encoder_pandas():
    dataframe = pd.DataFrame({"a": [1, 2, 3]})
    assert dumps(dataframe) == '{"a":{"0":1,"1":2,"2":3}}'

    series = pd.Series([1, 2, 3])
    assert dumps(series) == '{"0":1,"1":2,"2":3}'


def test_assert_pydantic_model():
    with pytest.raises(AssertionError):
        _ = JSON(pydantic_model=ExampleAttrsClass)  # type: ignore (testing pydantic check)


def test_warning_capture(caplog: LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        _ = JSON(pydantic_model=BaseSchema, validate_json=True)
    assert "has been deprecated" in caplog.text


def test_json_openapi_schema():
    assert JSON().openapi_schema().type == "object"

    schema = JSON(pydantic_model=BaseSchema).openapi_schema()
    assert schema.type == "object"
    assert schema.required == ["name", "endpoints"]
    assert schema.properties == {
        "name": {"title": "Name", "type": "string"},
        "endpoints": {
            "title": "Endpoints",
            "type": "array",
            "items": {"type": "string"},
        },
    }


def test_json_openapi_components():
    assert JSON().openapi_components() == {}

    components = JSON(pydantic_model=BaseSchema).openapi_components()

    assert components

    schema: Schema = components["schemas"]["BaseSchema"]

    assert schema.properties == {
        "name": {"title": "Name", "type": "string"},
        "endpoints": {
            "title": "Endpoints",
            "type": "array",
            "items": {"type": "string"},
        },
    }
    assert schema.type == "object"
    assert schema.required == ["name", "endpoints"]

    nested = JSON(pydantic_model=Nested).openapi_components()

    assert nested

    assert all(a in nested["schemas"] for a in ["BaseSchema", "Nested"])
    assert "$ref" in nested["schemas"]["Nested"].properties["nested"]


def test_json_openapi_request_responses():
    request_body = JSON().openapi_request_body()
    assert request_body.required

    assert "application/json" in request_body.content

    responses = JSON().openapi_responses()

    assert responses.content

    assert "application/json" in responses.content
