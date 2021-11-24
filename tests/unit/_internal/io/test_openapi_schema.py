import typing as t

import numpy as np
import pydantic
import pytest

from bentoml.io import JSON, File, Image, Multipart, NumpyNdarray, PandasDataFrame, Text


class _Schema(pydantic.BaseModel):
    name: str
    endpoints: t.List[str]


@pytest.mark.parametrize(
    "exp, model",
    [
        (
            {
                "application/json": {
                    "schema": {
                        "title": "_Schema",
                        "type": "object",
                        "properties": {
                            "name": {"title": "Name", "type": "string"},
                            "endpoints": {
                                "title": "Endpoints",
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name", "endpoints"],
                    }
                }
            },
            _Schema,
        ),
        ({"application/json": {"schema": {"type": "object"}}}, None),
    ],
)
def test_json_openapi_schema(
    exp: t.Dict[str, t.Any], model: t.Optional[pydantic.BaseModel]
):
    assert JSON(pydantic_model=model).openapi_request_schema() == exp
    assert JSON(pydantic_model=model).openapi_responses_schema() == exp


@pytest.mark.parametrize(
    "exp, mime_type",
    [
        (
            {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            None,
        ),
        (
            {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
            "application/pdf",
        ),
    ],
)
def test_file_openapi_schema(exp: t.Dict[str, t.Any], mime_type: str):
    assert File(mime_type=mime_type).openapi_request_schema() == exp
    assert File(mime_type=mime_type).openapi_responses_schema() == exp


@pytest.mark.parametrize(
    "exp, mime_type",
    [
        (
            {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            "image/jpeg",
        ),
        (
            {"image/png": {"schema": {"type": "string", "format": "binary"}}},
            "image/png",
        ),
    ],
)
def test_image_openapi_schema(exp: t.Dict[str, t.Any], mime_type: str):
    assert Image(mime_type=mime_type).openapi_request_schema() == exp
    assert Image(mime_type=mime_type).openapi_responses_schema() == exp


def test_text_openapi_schema():
    exp = {"text/plain": {"schema": {"type": "string"}}}
    assert Text().openapi_request_schema() == exp
    assert Text().openapi_responses_schema() == exp


@pytest.mark.parametrize(
    "exp, kwargs",
    [
        (
            {"text/csv": {"schema": {"type": "string", "format": "binary"}}},
            {"mime_type": "text/csv"},
        ),
        (
            {"application/json": {"schema": {"type": "object"}}},
            {"mime_type": "application/json", "dtype": "int"},
        ),
        (
            {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "array", "items": {"type": "integer"}},
                            "name": {"type": "array", "items": {"type": "string"}},
                        },
                    }
                }
            },
            {
                "mime_type": "application/json",
                "dtype": {"index": "int64", "name": "str"},
                "columns": ["int64", "str"],
            },
        ),
    ],
)
def test_pandas_openapi_schema(exp, kwargs):
    assert PandasDataFrame(**kwargs).openapi_request_schema() == exp
    assert PandasDataFrame(**kwargs).openapi_responses_schema() == exp


@pytest.mark.parametrize(
    "exp, kwargs",
    [
        ({"application/json": {"schema": {"type": "array", "items": {}}}}, {}),
        (
            {
                "application/json": {
                    "schema": {"type": "array", "items": {"type": "object"}}
                }
            },
            {"shape": (1,)},
        ),
        (
            {
                "application/json": {
                    "schema": {"type": "array", "items": {"type": "number"}}
                }
            },
            {"shape": (1,), "dtype": "complex128"},
        ),
        (
            {
                "application/json": {
                    "schema": {"type": "array", "items": {"type": "number"}}
                }
            },
            {"shape": (1,), "dtype": np.dtype("float32")},
        ),
        (
            {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                    }
                }
            },
            {"shape": (3, 2), "dtype": np.dtype("int8")},
        ),
    ],
)
def test_numpy_openapi_schema(exp, kwargs):
    assert NumpyNdarray(**kwargs).openapi_request_schema() == exp
    assert NumpyNdarray(**kwargs).openapi_responses_schema() == exp


def test_multipart_openapi_schema():
    array = NumpyNdarray(shape=(3, 2), dtype=np.dtype("float32"))
    dataframe = PandasDataFrame(
        dtype={"index": "int64", "name": "str"},
        columns=["index", "name"],
        orient="records",
    )
    exp = {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "arr": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "df": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "array", "items": {"type": "integer"}},
                            "name": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            }
        }
    }
    assert Multipart(arr=array, df=dataframe).openapi_request_schema() == exp
    assert Multipart(arr=array, df=dataframe).openapi_responses_schema() == exp
