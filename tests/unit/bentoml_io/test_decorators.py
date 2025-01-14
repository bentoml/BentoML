from typing import Generator

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from starlette.testclient import TestClient
from typing_extensions import Annotated

import bentoml
from bentoml.validators import DataframeSchema
from bentoml.validators import DType
from bentoml.validators import Shape


def test_mount_asgi_app():
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.asgi_app(app, path="/test")
    @bentoml.service(metrics={"enabled": False})
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    with TestClient(app=TestService.to_asgi()) as client:
        response = client.get("/test/hello")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, world!"


def test_mount_asgi_app_later():
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.service(metrics={"enabled": False})
    @bentoml.asgi_app(app, path="/test")
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    with TestClient(app=TestService.to_asgi()) as client:
        response = client.get("/test/hello")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, world!"


def test_service_instantiate():
    @bentoml.service
    class TestService:
        @bentoml.api
        def hello(self, name: str) -> str:
            return f"Hello, {name}!"

        @bentoml.api
        def stream(self, name: str) -> Generator[str, None, None]:
            for i in range(3):
                yield f"Hello, {name}! {i}"

    svc = TestService()
    assert svc.hello("world") == "Hello, world!"
    assert list(svc.stream("world")) == [
        "Hello, world! 0",
        "Hello, world! 1",
        "Hello, world! 2",
    ]


@pytest.mark.asyncio
async def test_service_instantiate_to_async():
    @bentoml.service
    class TestService:
        @bentoml.api
        def hello(self, name: str) -> str:
            return f"Hello, {name}!"

        @bentoml.api
        def stream(self, name: str) -> Generator[str, None, None]:
            for i in range(3):
                yield f"Hello, {name}! {i}"

    svc = TestService()
    assert await svc.to_async.hello("world") == "Hello, world!"
    assert [text async for text in svc.to_async.stream("world")] == [
        "Hello, world! 0",
        "Hello, world! 1",
        "Hello, world! 2",
    ]


def test_api_decorator_numpy():
    @bentoml.api
    def numpy_func(
        _,  # The decorator assumes `self` is the first arg.
        arr: npt.NDArray[np.float64],
    ) -> Annotated[npt.NDArray[np.int64], DType("int64"), Shape((1,))]:
        return arr.astype(np.int64)

    numpy_func.input_spec.model_fields["arr"].annotation is npt.NDArray[np.float64]
    numpy_func.output_spec.model_fields["root"].annotation is Annotated[
        npt.NDArray[np.int64], DType("int64"), Shape((1,))
    ]

    with pytest.raises(
        TypeError,
        match=r"Unable to infer the output spec for function .+, please specify output_spec manually",
    ):

        @bentoml.api
        def numpy_func(
            _,  # The decorator assumes `self` is the first arg.
            arr: npt.NDArray[np.float64],
        ) -> Annotated[npt.NDArray[np.float64], DType("int64"), Shape((1,))]:
            return arr.astype(np.int64)


def test_api_decorator_pandas():
    @bentoml.api
    def pandas_func(
        _,  # The decorator assumes `self` is the first arg.
        df1: pd.DataFrame,
        df2: Annotated[pd.DataFrame, DataframeSchema(columns=("b",))],
    ) -> Annotated[
        pd.DataFrame,
        DataframeSchema(orient="columns", columns=["a", "b"]),
    ]:
        return pd.concat([df1, df2], axis=1)

    pandas_func.input_spec.model_fields["df1"].annotation is pd.DataFrame
    pandas_func.input_spec.model_fields["df2"].annotation is Annotated[
        pd.DataFrame,
        DataframeSchema(columns=("b",)),
    ]
    pandas_func.output_spec.model_fields["root"].annotation is Annotated[
        pd.DataFrame,
        DataframeSchema(orient="columns", columns=("a", "b")),
    ]


def test_api_decorator_openapi_overrides():
    from bentoml._internal.service.factory import Service
    from bentoml._internal.service.openapi import generate_spec
    from bentoml._internal.service.openapi.specification import OpenAPISpecification

    @bentoml.service(name="test_overriden_service")
    class TestOverridenService(Service):
        @bentoml.api(
            openapi_overrides={
                "description": "My custom description",
                "tags": ["override-tag"],
                "parameters": [
                    {
                        "name": "version",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string"},
                    }
                ],
            }
        )
        def predict(self, data: str) -> str:
            return data

    svc = TestOverridenService()
    openapi_spec: OpenAPISpecification = generate_spec(svc)
    spec_dict = openapi_spec.asdict()

    # Verify custom description is included
    predict_operation = spec_dict["paths"]["/predict"]["post"]
    assert predict_operation["description"] == "My custom description"

    # Verify custom tags are included
    assert "override-tag" in predict_operation["tags"]

    # Verify custom parameters are included
    parameters = predict_operation["parameters"]
    version_param = next(
        (param for param in parameters if param["name"] == "version"), None
    )
    assert version_param is not None
    assert version_param["in"] == "query"
    assert version_param["required"] is False
    assert version_param["schema"]["type"] == "string"


def test_api_decorator_parameter_overrides():
    from bentoml._internal.service.factory import Service
    from bentoml._internal.service.openapi import generate_spec
    from bentoml._internal.service.openapi.specification import OpenAPISpecification

    @bentoml.service(name="test_parameter_service")
    class TestParameterService(Service):
        @bentoml.api(
            openapi_overrides={
                "parameters": [
                    {
                        "name": "filter",
                        "in": "query",
                        "required": True,
                        "schema": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    {
                        "name": "order",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["asc", "desc"]},
                        "description": "Sort order",
                    },
                ]
            }
        )
        def list_items(self, data: str) -> str:
            return data

    svc = TestParameterService()
    openapi_spec: OpenAPISpecification = generate_spec(svc)
    spec_dict = openapi_spec.asdict()

    # Get operation parameters
    operation = spec_dict["paths"]["/list_items"]["post"]
    parameters = operation["parameters"]

    # Verify filter parameter
    filter_param = next(param for param in parameters if param["name"] == "filter")
    assert filter_param["in"] == "query"
    assert filter_param["required"] is True
    assert filter_param["schema"]["type"] == "array"
    assert filter_param["schema"]["items"]["type"] == "string"
    assert filter_param["schema"]["minItems"] == 1

    # Verify order parameter
    order_param = next(param for param in parameters if param["name"] == "order")
    assert order_param["in"] == "query"
    assert order_param["schema"]["type"] == "string"
    assert order_param["schema"]["enum"] == ["asc", "desc"]
    assert order_param["description"] == "Sort order"


def test_api_decorator_response_overrides():
    from bentoml._internal.service.factory import Service
    from bentoml._internal.service.openapi import generate_spec
    from bentoml._internal.service.openapi.specification import OpenAPISpecification

    @bentoml.service(name="test_response_service")
    class TestResponseService(Service):
        @bentoml.api(
            openapi_overrides={
                "responses": {
                    "200": {
                        "description": "Custom success response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "data": {"type": "string"},
                                        "metadata": {
                                            "type": "object",
                                            "properties": {
                                                "timestamp": {"type": "string"},
                                                "version": {"type": "string"},
                                            },
                                            "required": ["timestamp"],
                                        },
                                    },
                                    "required": ["data", "metadata"],
                                }
                            }
                        },
                    },
                    "429": {
                        "description": "Rate limit exceeded",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "error": {"type": "string"},
                                        "retry_after": {"type": "integer"},
                                    },
                                }
                            }
                        },
                    },
                }
            }
        )
        def process_data(self, data: str) -> str:
            return data

    svc = TestResponseService()
    openapi_spec: OpenAPISpecification = generate_spec(svc)
    spec_dict = openapi_spec.asdict()

    # Get operation responses
    operation = spec_dict["paths"]["/process_data"]["post"]
    responses = operation["responses"]

    # Verify success response override
    success_response = responses["200"]
    assert success_response["description"] == "Custom success response"

    success_schema = success_response["content"]["application/json"]["schema"]
    assert success_schema["type"] == "object"
    assert "data" in success_schema["properties"]
    assert "metadata" in success_schema["properties"]
    assert success_schema["required"] == ["data", "metadata"]

    metadata_schema = success_schema["properties"]["metadata"]
    assert metadata_schema["type"] == "object"
    assert "timestamp" in metadata_schema["properties"]
    assert "version" in metadata_schema["properties"]
    assert metadata_schema["required"] == ["timestamp"]

    # Verify custom error response
    rate_limit_response = responses["429"]
    assert rate_limit_response["description"] == "Rate limit exceeded"

    error_schema = rate_limit_response["content"]["application/json"]["schema"]
    assert error_schema["type"] == "object"
    assert "error" in error_schema["properties"]
    assert "retry_after" in error_schema["properties"]
    assert error_schema["properties"]["retry_after"]["type"] == "integer"


def test_api_decorator_multiple_overrides():
    from bentoml._internal.service import Service
    from bentoml._internal.service.openapi import generate_spec
    from bentoml._internal.service.openapi.specification import OpenAPISpecification

    @bentoml.service(name="test_multi_endpoint_service")
    class TestMultiEndpointService(Service):
        def __init__(self):
            super().__init__(
                config={"name": "test_multi_endpoint_service"}, inner=self.__class__
            )

        @bentoml.api(
            openapi_overrides={
                "description": "First endpoint description",
                "tags": ["tag-1"],
                "parameters": [
                    {"name": "param1", "in": "query", "schema": {"type": "string"}}
                ],
            }
        )
        def endpoint_one(self, data: str) -> str:
            return data

        @bentoml.api(
            openapi_overrides={
                "description": "Second endpoint description",
                "tags": ["tag-2"],
                "responses": {
                    "200": {
                        "description": "Success response for endpoint two",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "result": {"type": "string"},
                                        "status": {"type": "string"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        )
        def endpoint_two(self, data: str) -> str:
            return data

    svc = TestMultiEndpointService()
    openapi_spec: OpenAPISpecification = generate_spec(svc)
    spec_dict = openapi_spec.asdict()

    # Verify endpoint one overrides
    endpoint_one = spec_dict["paths"]["/endpoint_one"]["post"]
    assert endpoint_one["description"] == "First endpoint description"
    assert "tag-1" in endpoint_one["tags"]
    parameters = endpoint_one["parameters"]
    param1 = next(param for param in parameters if param["name"] == "param1")
    assert param1["in"] == "query"
    assert param1["schema"]["type"] == "string"

    # Verify endpoint two overrides
    endpoint_two = spec_dict["paths"]["/endpoint_two"]["post"]
    assert endpoint_two["description"] == "Second endpoint description"
    assert "tag-2" in endpoint_two["tags"]
    success_response = endpoint_two["responses"]["200"]
    assert success_response["description"] == "Success response for endpoint two"
    response_schema = success_response["content"]["application/json"]["schema"]
    assert "result" in response_schema["properties"]
    assert "status" in response_schema["properties"]

    # Verify no cross-contamination
    assert "tag-1" not in endpoint_two["tags"]
    assert "tag-2" not in endpoint_one["tags"]
    assert "param1" not in [p.get("name") for p in endpoint_two.get("parameters", [])]


def test_api_decorator_invalid_overrides():
    """Test that invalid OpenAPI overrides raise appropriate errors."""
    import typing as t

    import pytest

    from bentoml._internal.service.factory import Service
    from bentoml._internal.service.openapi import generate_spec

    # Test invalid field name
    @bentoml.service(name="test_invalid_field_service")
    class TestInvalidFieldService(Service):
        @bentoml.api(
            openapi_overrides={
                "invalid_field": "some value",  # Invalid OpenAPI field name
            }
        )
        def predict(self: t.Any, data: str) -> str:
            return data

    svc_field = TestInvalidFieldService()
    with pytest.raises(ValueError, match="Invalid OpenAPI field"):
        generate_spec(svc_field)

    # Test invalid field value type
    @bentoml.service(name="test_invalid_value_service")
    class TestInvalidValueService(Service):
        @bentoml.api(
            openapi_overrides={
                "parameters": "not a list",  # Parameters must be a list
            }
        )
        def predict(self: t.Any, data: str) -> str:
            return data

    svc_value = TestInvalidValueService()
    with pytest.raises(TypeError, match="Invalid value type"):
        generate_spec(svc_value)

    # Test invalid nested schema
    @bentoml.service(name="test_invalid_schema_service")
    class TestInvalidSchemaService(Service):
        @bentoml.api(
            openapi_overrides={
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "invalid_type",  # Invalid schema type
                                }
                            }
                        }
                    }
                }
            }
        )
        def predict(self: t.Any, data: str) -> str:
            return data

    svc_schema = TestInvalidSchemaService()
    with pytest.raises(ValueError, match="Invalid schema"):
        generate_spec(svc_schema)

    # Test invalid response code
    @bentoml.service(name="test_invalid_response_service")
    class TestInvalidResponseService(Service):
        @bentoml.api(
            openapi_overrides={
                "responses": {
                    "999": {  # Invalid HTTP status code
                        "description": "Invalid response"
                    }
                }
            }
        )
        def predict(self: t.Any, data: str) -> str:
            return data

    svc_response = TestInvalidResponseService()
    with pytest.raises(ValueError, match="Invalid response code"):
        generate_spec(svc_response)
