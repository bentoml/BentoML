from __future__ import annotations

import sys
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
from bentoml.exceptions import BadInput
from bentoml.exceptions import UnprocessableEntity
from bentoml.grpc.utils import import_generated_stubs
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils.pkg import pkg_version_info
from bentoml._internal.io_descriptors.json import DefaultJsonEncoder

if sys.version_info >= (3, 11):
    from typing import Required
    from typing import TypedDict
    from typing import NotRequired

else:
    from typing_extensions import Required
    from typing_extensions import TypedDict
    from typing_extensions import NotRequired


if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from google.protobuf import struct_pb2

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml._internal.service.openapi.specification import Schema
else:
    pb, _ = import_generated_stubs()
    struct_pb2 = LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")


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


@pytest.mark.skipif(
    pkg_version_info("pydantic")[0] < 2 and pkg_version_info("bentoml")[:2] <= (1, 1),
    reason="Pydantic 2.x is not yet supported until official releases of Pydantic.",
)
def test_not_yet_supported_pydantic():
    with pytest.raises(UnprocessableEntity) as exc_info:
        JSON(pydantic_model=Nested)
    assert "pydantic 2.x is not yet supported" in str(exc_info.value)


def test_invalid_init():
    with pytest.raises(AssertionError) as exc_info:
        JSON(pydantic_model=ExampleAttrsClass)  # type: ignore (testing exception)
    assert "'pydantic_model' must be a subclass of 'pydantic.BaseModel'." in str(
        exc_info.value
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
    assert request_body["required"]

    assert "application/json" in request_body["content"]

    responses = JSON().openapi_responses()

    assert responses["content"]

    assert "application/json" in responses["content"]


@pytest.mark.asyncio
async def test_from_proto():
    res = await JSON().from_proto(
        b'{"request_id": "123", "iris_features": {"sepal_len":2.34,"sepal_width":1.58, "petal_len":6.52, "petal_width":3.23}}',
    )
    assert res == {
        "request_id": "123",
        "iris_features": {
            "sepal_len": 2.34,
            "sepal_width": 1.58,
            "petal_len": 6.52,
            "petal_width": 3.23,
        },
    }
    res = await JSON(pydantic_model=BaseSchema).from_proto(
        b'{"name":"test","endpoints":["predict","health"]}',
    )
    assert isinstance(res, pydantic.BaseModel) and res == BaseSchema(
        name="test", endpoints=["predict", "health"]
    )
    res = await JSON(pydantic_model=Nested).from_proto(
        struct_pb2.Value(
            struct_value=struct_pb2.Struct(
                fields={
                    "toplevel": struct_pb2.Value(string_value="test"),
                    "nested": struct_pb2.Value(
                        struct_value=struct_pb2.Struct(
                            fields={
                                "name": struct_pb2.Value(string_value="test"),
                                "endpoints": struct_pb2.Value(
                                    list_value=struct_pb2.ListValue(
                                        values=[
                                            struct_pb2.Value(string_value="predict"),
                                            struct_pb2.Value(string_value="health"),
                                        ]
                                    )
                                ),
                            }
                        ),
                    ),
                }
            )
        ),
    )
    assert isinstance(res, pydantic.BaseModel) and res == Nested(
        toplevel="test",
        nested=BaseSchema(name="test", endpoints=["predict", "health"]),
    )


@pytest.mark.asyncio
async def test_exception_from_proto():
    with pytest.raises(AssertionError):
        await JSON().from_proto(pb.NDArray(string_values="asdf"))  # type: ignore (testing exception)
        await JSON().from_proto("")  # type: ignore (testing exception)
    with pytest.raises(BadInput, match="Invalid JSON input received*"):
        await JSON(pydantic_model=Nested).from_proto(
            struct_pb2.Value(string_value="asdf")
        )
    with pytest.raises(BadInput, match="Invalid JSON input received*"):
        await JSON(pydantic_model=Nested).from_proto(b"")
        await JSON().from_proto(b"\n?xfa")


@pytest.mark.asyncio
async def test_exception_to_proto():
    with pytest.raises(TypeError):
        await JSON().to_proto(b"asdf")  # type: ignore (testing exception)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "o",
    [
        {"asdf": 1},
        ["asdf", "1"],
        "asdf",
        1.0,
        1,
        True,
        BaseSchema(name="test", endpoints=["predict", "health"]),
        np.random.rand(6, 6),
        None,
    ],
)
async def test_to_proto(o: t.Any) -> None:
    res = await JSON().to_proto(o)
    assert res and isinstance(res, struct_pb2.Value)


class InnerTypedDict(TypedDict):
    str1: str
    any1: t.Any


class NestedTypedDict(TypedDict):
    num1: float
    num2: int
    str1: str
    nested_typeddict1: InnerTypedDict


class ExampleTypedDictSchema(TypedDict, total=False):
    num1: float
    num2: int
    str1: str
    dict1: t.Dict[str, int]
    tuple1: t.Tuple[int, str, bool]
    set1: t.Set[int]
    list1: t.List[str]
    union1: t.Union[int, float]
    required_num: Required[float]
    notrequired_str: NotRequired[str]
    any1: t.Any


class ExamplePydanticSchema(BaseSchema):
    num1: float
    num2: int
    str1: str
    dict1: t.Dict[str, int]
    tuple1: t.Tuple[int, str, bool]
    set1: t.Set[int]
    list1: t.List[str]
    union1: t.Union[int, float]
    required_num: float
    notrequired_str: t.Optional[str]


def test_exception_typeddict():
    with pytest.raises(BadInput, match="'typeddict' must be inherited 'TypedDict'."):
        _ = JSON(typeddict=ExampleAttrsClass)


def test_exception_typeddict_and_pydantic_model_are_mutually_exclusive():
    with pytest.raises(
        BadInput,
        match="'typeddict' and 'pydantic_model' are mutually exclusive. Make sure to only use one of them.",
    ):
        _ = JSON(typeddict=ExampleTypedDictSchema, pydantic_model=ExamplePydanticSchema)


def test_json_typeddict_openapi_schema():
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


def test_typeddict_json_openapi_components():
    assert JSON().openapi_components() == {}

    components = JSON(typeddict=ExampleTypedDictSchema).openapi_components()

    assert components

    schema: Schema = components["schemas"][f"{__name__}__ExampleTypedDictSchema"]

    assert schema.properties == {
        "any1": {"title": "any1", "type": "any"},
        "dict1": {
            "additionalProperties": {"type": "integer"},
            "title": "dict1",
            "type": "object",
        },
        "list1": {"items": {"type": "string"}, "title": "list1", "type": "array"},
        "notrequired_str": {"title": "notrequired_str", "type": "string"},
        "num1": {"title": "num1", "type": "number"},
        "num2": {"title": "num2", "type": "integer"},
        "required_num": {"title": "required_num", "type": "number"},
        "set1": {"items": {"type": "integer"}, "title": "set1", "type": "array"},
        "str1": {"title": "str1", "type": "string"},
        "tuple1": {
            "items": [{"type": "integer"}, {"type": "string"}, {"type": "boolean"}],
            "title": "tuple1",
            "type": "array",
        },
        "union1": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
    }
    assert schema.type == "object"
    assert schema.required == [
        "num1",
        "num2",
        "str1",
        "dict1",
        "tuple1",
        "set1",
        "list1",
        "union1",
        "required_num",
        "any1",
    ]

    nested = JSON(typeddict=NestedTypedDict).openapi_components()

    assert nested

    assert all(
        a in nested["schemas"]
        for a in [f"{__name__}__InnerTypedDict", f"{__name__}__NestedTypedDict"]
    )
    assert (
        "$ref"
        in nested["schemas"][f"{__name__}__NestedTypedDict"].properties[
            "nested_typeddict1"
        ]
    )


def test_typeddict_json_openapi_request_responses():
    request_body = JSON(typeddict=ExampleTypedDictSchema).openapi_request_body()

    assert request_body["required"]

    assert (
        "notrequired_str"
        not in request_body["content"]["application/json"].schema.required
    )

    assert "application/json" in request_body["content"]

    responses = JSON().openapi_responses()

    assert responses["content"]

    assert "application/json" in responses["content"]
