from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from starlette.requests import Request

from bentoml._internal.utils.lazy_loader import LazyLoader
from bentoml.exceptions import BadInput
from bentoml.grpc.utils import import_generated_stubs
from bentoml.io import Multipart
from bentoml.io import Text

if TYPE_CHECKING:
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1 import service_pb2 as pb
else:
    pb, _ = import_generated_stubs()
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")


def test_text_openapi_schema():
    assert Text().openapi_schema().type == "string"


def test_init():
    assert Text().mime_type == "text/plain"
    assert Text(content_type="text/event-stream").mime_type == "text/event-stream"


def test_text_openapi_request_responses():
    mime_type = "text/plain"

    request_body = Text().openapi_request_body()
    assert request_body["required"]

    assert mime_type in request_body["content"]

    responses = Text().openapi_responses()

    assert responses["content"]

    assert mime_type in responses["content"]


@pytest.mark.asyncio
async def test_from_proto():
    res = await Text().from_proto(wrappers_pb2.StringValue(value="asdf"))
    assert res == "asdf"
    res = await Text().from_proto(b"asdf")
    assert res == "asdf"


@pytest.mark.asyncio
async def test_exception_from_proto():
    with pytest.raises(AssertionError):
        await Text().from_proto(pb.NDArray(string_values="asdf"))  # type: ignore (testing exception)
        await Text().from_proto(b"")


@pytest.mark.asyncio
async def test_to_proto() -> None:
    res = await Text().to_proto("asdf")
    assert res.value == "asdf"


def _make_request(body: bytes, content_type: str) -> Request:
    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = False
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", content_type.encode("latin-1"))],
        "path": "/",
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
        "client": ("testclient", 50000),
    }
    return Request(scope, receive=receive)


@pytest.mark.asyncio
async def test_from_http_request_non_utf8_raises_bad_input():
    # bentoml/BentoML#5781: a body that is not UTF-8 must be a 400, not a 500.
    with pytest.raises(BadInput):
        await Text().from_http_request(_make_request(b"\xa9", "text/plain"))


@pytest.mark.asyncio
async def test_from_proto_non_utf8_raises_bad_input():
    with pytest.raises(BadInput):
        await Text().from_proto(b"\xa9")


@pytest.mark.asyncio
async def test_multipart_non_utf8_file_part_raises_bad_input():
    # The same non-UTF-8 body reached through Multipart must also be a 400.
    boundary = "----testboundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="text"; filename="t.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n".encode("latin-1")
        + b"\xa9"
        + f"\r\n--{boundary}--\r\n".encode("latin-1")
    )
    request = _make_request(body, f"multipart/form-data; boundary={boundary}")
    with pytest.raises(BadInput):
        await Multipart(text=Text()).from_http_request(request)
