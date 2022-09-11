from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bentoml.io import Text
from bentoml.exceptions import BentoMLException

if TYPE_CHECKING:
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs
    from bentoml._internal.utils import LazyLoader

    pb, _ = import_generated_stubs()
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")


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


@pytest.mark.asyncio
async def test_from_proto(use_internal_bytes_contents: bool):
    if not use_internal_bytes_contents:
        res = await Text().from_proto(wrappers_pb2.StringValue(value="asdf"))
        assert res == "asdf"
        res = await Text().from_proto(
            pb.Part(text=wrappers_pb2.StringValue(value="asdf")),
            _use_internal_bytes_contents=use_internal_bytes_contents,
        )
        assert res == "asdf"
    else:
        res = await Text().from_proto(
            b"asdf",
            _use_internal_bytes_contents=use_internal_bytes_contents,
        )
        assert res == "asdf"


@pytest.mark.asyncio
async def test_exception_from_proto(use_internal_bytes_contents: bool):
    with pytest.raises(AssertionError):
        await Text().from_proto(
            pb.NDArray(string_values="asdf"),
            _use_internal_bytes_contents=use_internal_bytes_contents,
        )
        await Text().from_proto(
            "", _use_internal_bytes_contents=use_internal_bytes_contents
        )


@pytest.mark.asyncio
async def test_to_proto() -> None:
    res = await Text().to_proto("asdf")
    assert res.value == "asdf"
