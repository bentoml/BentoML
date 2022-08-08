from __future__ import annotations

import pytest

from bentoml.io import File


def test_file_openapi_schema():
    assert File().openapi_schema().type == "string"
    assert File().openapi_schema().format == "binary"


def test_invalid_kind():
    with pytest.raises(ValueError):
        _ = File(kind="asdf")  # type: ignore (testing error handling)


@pytest.mark.parametrize("mime_type", ["application/octet-stream", "application/pdf"])
def test_file_openapi_request_responses(mime_type: str):
    request_body = File(mime_type=mime_type).openapi_request_body()
    assert request_body.required

    assert mime_type in request_body.content

    responses = File(mime_type=mime_type).openapi_responses()

    assert responses.content

    assert mime_type in responses.content
