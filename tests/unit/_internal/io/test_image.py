from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from bentoml.io import Image
from bentoml.exceptions import BadInput
from bentoml.exceptions import InvalidArgument
from bentoml.grpc.utils import import_generated_stubs
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image as PILImage

    from bentoml.grpc.v1 import service_pb2 as pb
else:
    pb, _ = import_generated_stubs()
    np = LazyLoader("np", globals(), "numpy")
    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")


def test_invalid_init():
    with pytest.raises(InvalidArgument) as exc_info:
        Image(mime_type="application/vnd.bentoml+json")
    assert "Invalid Image mime_type" in str(exc_info.value)
    with pytest.raises(InvalidArgument) as exc_info:
        Image(pilmode="asdf")
    assert "Invalid Image pilmode" in str(exc_info.value)


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
    assert request_body["required"]

    assert mime_type in request_body["content"]

    responses = Image(mime_type=mime_type).openapi_responses()

    assert responses["content"]

    assert mime_type in responses["content"]


@pytest.mark.asyncio
async def test_from_proto(img_file: str):
    with open(img_file, "rb") as f:
        content = f.read()
    res = await Image(mime_type="image/bmp").from_proto(content)
    assert_file = PILImage.open(img_file)
    np.testing.assert_array_almost_equal(np.array(res), np.array(assert_file))


@pytest.mark.asyncio
async def test_exception_from_proto():
    with pytest.raises(AssertionError):
        await Image().from_proto(pb.NDArray(string_values="asdf"))  # type: ignore (testing exception)
        await Image().from_proto("")  # type: ignore (testing exception)
    with pytest.raises(BadInput) as exc_info:
        await Image(mime_type="image/jpeg").from_proto(
            pb.File(kind="application/octet-stream", content=b"asdf")
        )
    assert "MIME type from 'kind'" in str(exc_info.value)
    with pytest.raises(BadInput) as exc_info:
        await Image(mime_type="image/jpeg").from_proto(pb.File(kind="image/jpeg"))
    assert "Content is empty!" == str(exc_info.value)


@pytest.mark.asyncio
async def test_exception_to_proto():
    with pytest.raises(BadInput) as exc_info:
        await Image().to_proto(io.BytesIO(b"asdf"))  # type: ignore (testing exception)
    assert "Unsupported Image type received:" in str(exc_info.value)


@pytest.mark.asyncio
async def test_to_proto(img_file: str) -> None:
    with open(img_file, "rb") as f:
        content = f.read()
    img = PILImage.open(io.BytesIO(content))
    res = await Image(mime_type="image/bmp").to_proto(img)
    assert res.kind == "image/bmp"
