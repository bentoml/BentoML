from __future__ import annotations

import pytest

from bentoml.io import Image
from bentoml.exceptions import InvalidArgument


def test_image_openapi_schema():
    assert Image().openapi_schema().type == "string"
    assert Image().openapi_schema().format == "binary"


def test_invalid_pilmode():
    with pytest.raises(InvalidArgument):
        _ = Image(pilmode="asdf")  # type: ignore (testing exception)

    with pytest.raises(InvalidArgument):
        _ = Image(mime_type="asdf")


@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg"])
def test_image_openapi_request_responses(mime_type: str):
    request_body = Image(mime_type=mime_type).openapi_request_body()
    assert request_body.required

    assert mime_type in request_body.content

    responses = Image(mime_type=mime_type).openapi_responses()

    assert responses.content

    assert mime_type in responses.content
