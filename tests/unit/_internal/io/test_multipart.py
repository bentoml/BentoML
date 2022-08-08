from __future__ import annotations

import pytest

from bentoml.io import JSON
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.exceptions import InvalidArgument

multipart = Multipart(arg1=JSON(), arg2=Image(pilmode="RGB"))


def test_invalid_multipart():
    with pytest.raises(InvalidArgument):
        _ = Multipart(arg1=Multipart(arg1=JSON()))


def test_multipart_openapi_schema():
    schema = multipart.openapi_schema()
    assert schema.type == "object"

    assert schema.properties
    assert all(arg in schema.properties for arg in ["arg1", "arg2"])


def test_multipart_openapi_request_responses():
    request_body = multipart.openapi_request_body()
    assert request_body.required

    responses = multipart.openapi_responses()

    assert responses.content
