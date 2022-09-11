from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bentoml.io import JSON
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.exceptions import InvalidArgument

multipart = Multipart(arg1=JSON(), arg2=Image(pilmode="RGB"))

if TYPE_CHECKING:
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs
    from bentoml._internal.utils import LazyLoader

    pb, _ = import_generated_stubs()
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")


def test_invalid_multipart():
    with pytest.raises(
        InvalidArgument,
        match="Multipart IO can not contain nested Multipart IO descriptor",
    ):
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


@pytest.mark.asyncio
async def test_exception_from_to_proto():
    with pytest.raises(InvalidArgument):
        await multipart.from_proto(b"", _use_internal_bytes_contents=True)
    with pytest.raises(InvalidArgument) as e:
        await multipart.from_proto(
            {"asdf": pb.Part(text=wrappers_pb2.StringValue(value="asdf"))}
        )
    assert "as input fields. Invalid fields are: " in str(e.value)
    with pytest.raises(InvalidArgument) as e:
        await multipart.to_proto(
            {"asdf": pb.Part(text=wrappers_pb2.StringValue(value="asdf"))}
        )
    assert "as output fields. Invalid fields are: " in str(e.value)
