from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from bentoml.io import JSON
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.exceptions import InvalidArgument
from bentoml.grpc.utils import import_generated_stubs
from bentoml._internal.utils import LazyLoader

example = Multipart(arg1=JSON(), arg2=Image(mime_type="image/bmp", pilmode="RGB"))

if TYPE_CHECKING:
    import PIL.Image as PILImage
    from google.protobuf import struct_pb2
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1 import service_pb2 as pb
else:
    pb, _ = import_generated_stubs()
    np = LazyLoader("np", globals(), "numpy")
    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")
    struct_pb2 = LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")


def test_invalid_multipart():
    with pytest.raises(
        InvalidArgument,
        match="Multipart IO can not contain nested Multipart IO descriptor",
    ):
        _ = Multipart(arg1=Multipart(arg1=JSON()))


def test_multipart_openapi_schema():
    schema = example.openapi_schema()
    assert schema.type == "object"

    assert schema.properties
    assert all(arg in schema.properties for arg in ["arg1", "arg2"])


def test_multipart_openapi_request_responses():
    request_body = example.openapi_request_body()
    assert request_body["required"]

    responses = example.openapi_responses()

    assert responses["content"]


@pytest.mark.asyncio
async def test_exception_from_to_proto():
    with pytest.raises(InvalidArgument):
        await example.from_proto(b"")  # type: ignore (test exception)
    with pytest.raises(InvalidArgument) as e:
        await example.from_proto(
            pb.Multipart(
                fields={"asdf": pb.Part(text=wrappers_pb2.StringValue(value="asdf"))}
            )
        )
    assert f"'{example!r}' accepts the following keys: " in str(e.value)
    with pytest.raises(InvalidArgument) as e:
        await example.to_proto(
            {"asdf": pb.Part(text=wrappers_pb2.StringValue(value="asdf"))}
        )
    assert f"'{example!r}' accepts the following keys: " in str(e.value)


@pytest.mark.asyncio
async def test_multipart_from_to_proto(img_file: str):
    with open(img_file, "rb") as f:
        img = f.read()
    obj = await example.from_proto(
        pb.Multipart(
            fields={
                "arg1": pb.Part(
                    json=struct_pb2.Value(
                        struct_value=struct_pb2.Struct(
                            fields={"asd": struct_pb2.Value(string_value="asd")}
                        )
                    )
                ),
                "arg2": pb.Part(file=pb.File(kind="image/bmp", content=img)),
            }
        )
    )
    assert obj["arg1"] == {"asd": "asd"}
    assert_file = PILImage.open(img_file)
    np.testing.assert_array_almost_equal(np.array(obj["arg2"]), np.array(assert_file))

    message = await example.to_proto(
        {"arg1": {"asd": "asd"}, "arg2": PILImage.open(io.BytesIO(img))}
    )
    assert isinstance(message, pb.Multipart)
    assert message.fields["arg1"].json.struct_value.fields["asd"].string_value == "asd"
