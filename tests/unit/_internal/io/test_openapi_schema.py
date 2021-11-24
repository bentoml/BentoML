import typing as t

import pydantic
import pytest

from bentoml.io import JSON

from .test_json import _Schema


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
def test_openapi_schema(exp: t.Dict[str, t.Any], model: t.Optional[pydantic.BaseModel]):
    assert JSON(pydantic_model=model).openapi_request_schema() == exp
    assert JSON(pydantic_model=model).openapi_responses_schema() == exp
