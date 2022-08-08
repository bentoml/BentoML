from __future__ import annotations

import pytest

from bentoml.io import Text
from bentoml.exceptions import BentoMLException


def test_text_openapi_schema():
    assert Text().openapi_schema().type == "string"


def test_invalid_init():
    with pytest.raises(BentoMLException):
        _ = Text(mime_type="asdf")


def test_text_openapi_request_responses():
    mime_type = "text/plain"

    request_body = Text().openapi_request_body()
    assert request_body.required

    assert mime_type in request_body.content

    responses = Text().openapi_responses()

    assert responses.content

    assert mime_type in responses.content
