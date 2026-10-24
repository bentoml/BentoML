from __future__ import annotations

import typing as t

import numpy as np
import pytest
from pydantic import BaseModel
from pydantic import ConfigDict
from starlette.applications import Starlette

import bentoml
from _bentoml_impl.client.http import AsyncHTTPClient
from _bentoml_impl.serde import JSONSerde
from _bentoml_impl.serde import PickleSerde


@pytest.mark.parametrize("serde_type", [JSONSerde, PickleSerde])
@pytest.mark.parametrize("additional_properties", [True, False, {}, None])
def test_serde_additional_properties(
    serde_type: type[JSONSerde] | type[PickleSerde],
    additional_properties: bool | dict[str, t.Any] | None,
) -> None:
    serde = serde_type()
    schema: dict[str, t.Any] = {
        "type": "object",
        "properties": {"score": {"type": "number"}},
    }
    if additional_properties is not None:
        schema["additionalProperties"] = additional_properties
    value = {"score": 0.8, "model_version": "v2", "metadata": {"source": "test"}}
    expected = {"score": 0.8} if additional_properties is False else value

    encoded = serde.deserialize_value(serde.serialize(value, schema))
    assert encoded == expected
    assert serde.deserialize(serde.serialize_value(value), schema) == expected
    assert value == {
        "score": 0.8,
        "model_version": "v2",
        "metadata": {"source": "test"},
    }


@pytest.mark.parametrize("serde_type", [JSONSerde, PickleSerde])
def test_serde_nested_additional_properties(
    serde_type: type[JSONSerde] | type[PickleSerde],
) -> None:
    serde = serde_type()
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,
        },
    }
    value = [{"name": "first", "score": 0.8}, {"name": "second", "score": 0.9}]

    assert serde.deserialize_value(serde.serialize(value, schema)) == value
    assert serde.deserialize(serde.serialize_value(value), schema) == value


@pytest.mark.parametrize("serde_type", [JSONSerde, PickleSerde])
def test_serde_additional_property_schema(
    serde_type: type[JSONSerde] | type[PickleSerde],
) -> None:
    serde = serde_type()
    tensor_schema = {"type": "tensor", "format": "numpy-array", "dtype": "float32"}
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "scores": tensor_schema},
        "additionalProperties": tensor_schema,
    }
    value = {
        "name": "prediction",
        "scores": np.array([0.8], dtype="float32"),
        "embeddings": np.array([1, 2], dtype="float32"),
    }

    encoded = serde.deserialize_value(serde.serialize(value, schema))
    assert set(encoded) == set(value)
    decoded = serde.deserialize(serde.serialize_value(encoded), schema)
    assert decoded["name"] == value["name"]
    for field in ("scores", "embeddings"):
        np.testing.assert_array_equal(decoded[field], value[field])
        assert decoded[field].dtype == np.dtype("float32")


class _FlexibleOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    score: float


@pytest.mark.asyncio
async def test_schema_driven_client_preserves_extra_output_fields() -> None:
    @bentoml.service(metrics={"enabled": False})
    class PredictionService:
        @bentoml.api
        def predict(self) -> _FlexibleOutput:
            return _FlexibleOutput.model_validate({"score": 0.8, "model_version": "v2"})

    app = t.cast(Starlette, PredictionService.to_asgi())
    async with app.router.lifespan_context(app):
        async with AsyncHTTPClient("http://localhost", app=app) as client:
            result = await client.predict()

    assert result == {"score": 0.8, "model_version": "v2"}
