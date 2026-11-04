import sys
from typing import Any

import numpy as np
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from _bentoml_impl.client.http import AsyncHTTPClient
from _bentoml_impl.serde import JSONSerde
from _bentoml_sdk import validators
from bentoml._internal.utils.lazy_loader import LazyLoader


@pytest.mark.asyncio
@pytest.mark.parametrize("framework", ["torch", "tensorflow"])
@pytest.mark.parametrize("null_first", [False, True])
@pytest.mark.parametrize("value", [[1.0, 2.0], [[1.0], [2.0]], [], 1.0, None])
async def test_nullable_tensor_json_client_without_framework(
    framework: str, null_first: bool, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_name = "tf" if framework == "tensorflow" else "torch"
    await assert_nullable_json_client_without_framework(
        framework,
        local_name,
        {"type": "tensor", "format": f"{local_name}-tensor"},
        null_first,
        value,
        monkeypatch,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("null_first", [False, True])
@pytest.mark.parametrize(
    ("orient", "value"),
    [
        ("records", [{"a": 1}, {"a": 2}]),
        ("records", []),
        ("columns", {"a": [1, 2]}),
        ("columns", {}),
        ("records", None),
    ],
)
async def test_nullable_dataframe_json_client_without_pandas(
    orient: str, value: Any, null_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    await assert_nullable_json_client_without_framework(
        "pandas",
        "pd",
        {"type": "dataframe", "orient": orient},
        null_first,
        value,
        monkeypatch,
    )


async def assert_nullable_json_client_without_framework(
    framework: str,
    local_name: str,
    schema_branch: dict[str, str],
    null_first: bool,
    value: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, framework, None)
    loader = LazyLoader(local_name, {}, framework)
    monkeypatch.setattr(validators, local_name, loader)
    branches = [
        schema_branch,
        {"type": "null"},
    ]
    if null_first:
        branches.reverse()

    async def schema(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "routes": [
                    {
                        "name": "predict",
                        "route": "/predict",
                        "input": {"root_input": True, "anyOf": branches},
                        "output": {},
                    }
                ]
            }
        )

    received = []

    async def predict(request: Request) -> JSONResponse:
        body = await request.json()
        received.append(body)
        return JSONResponse(body)

    app = Starlette(
        routes=[
            Route("/schema.json", schema),
            Route("/predict", predict, methods=["POST"]),
        ]
    )
    async with AsyncHTTPClient("http://test", app=app) as client:
        assert await client.predict(value) == value
    assert received == [value]
    assert loader._module is None


@pytest.mark.parametrize("null_first", [False, True])
@pytest.mark.parametrize(
    "value", [np.array([1.0, 2.0]), [np.array(1.0), np.array(2.0)]]
)
def test_nullable_numpy_tensor_encoding(null_first: bool, value: Any) -> None:
    branches = [{"type": "tensor", "format": "numpy-array"}, {"type": "null"}]
    if null_first:
        branches.reverse()
    payload = JSONSerde().serialize(value, {"anyOf": branches})
    assert JSONSerde().deserialize_value(payload) == [1.0, 2.0]


@pytest.mark.parametrize(
    "branches",
    [
        [{"type": "tensor", "format": "torch-tensor"}, {"type": "string"}],
        [
            {"type": "tensor", "format": "torch-tensor"},
            {"type": "string"},
            {"type": "null"},
        ],
    ],
)
def test_general_union_encoding_unchanged(branches: list[dict[str, str]]) -> None:
    value = [1.0, 2.0]
    assert JSONSerde()._encode(value, {"anyOf": branches}) is value


def test_required_tensor_encoding_unchanged() -> None:
    payload = JSONSerde().serialize(
        [1.0, 2.0],
        {"type": "tensor", "format": "numpy-array", "shape": [2, 1]},
    )
    assert JSONSerde().deserialize_value(payload) == [[1.0], [2.0]]


@pytest.mark.parametrize("orient", ["records", "columns"])
def test_nullable_dataframe_encoding(orient: str) -> None:
    import pandas as pd

    frame = pd.DataFrame({"a": [1, 2]})
    payload = JSONSerde().serialize(
        frame, {"anyOf": [{"type": "dataframe", "orient": orient}, {"type": "null"}]}
    )
    expected = [{"a": 1}, {"a": 2}] if orient == "records" else {"a": [1, 2]}
    assert JSONSerde().deserialize_value(payload) == expected
