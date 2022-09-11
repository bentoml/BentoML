from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from bentoml.io import File
from bentoml.exceptions import BadInput

if TYPE_CHECKING:
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()


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


@pytest.mark.asyncio
async def test_from_proto(use_internal_bytes_contents: bool, bin_file: str):
    with open(bin_file, "rb") as f:
        content = f.read()
    if use_internal_bytes_contents:
        res = await File().from_proto(
            content, _use_internal_bytes_contents=use_internal_bytes_contents
        )
        assert res.read() == b"\x810\x899"
    else:
        res = await File().from_proto(
            pb.Part(file=pb.File(content=content)),
            _use_internal_bytes_contents=use_internal_bytes_contents,
        )
        assert res.read() == b"\x810\x899"


@pytest.mark.asyncio
async def test_exception_from_proto(use_internal_bytes_contents: bool):
    with pytest.raises(AssertionError):
        await File().from_proto(
            pb.NDArray(string_values="asdf"),
            _use_internal_bytes_contents=use_internal_bytes_contents,
        )
        await File().from_proto(
            "", _use_internal_bytes_contents=use_internal_bytes_contents
        )
    with pytest.raises(BadInput) as exc_info:
        await File(mime_type="multipart/form-data").from_proto(pb.File())
    assert "which is not allowed to use with 'from_proto'" in str(exc_info.value)
    if not use_internal_bytes_contents:
        with pytest.raises(BadInput) as exc_info:
            await File(mime_type="image/jpeg").from_proto(
                pb.File(kind=pb.File.FILE_TYPE_BYTES, content=b"asdf"),
                _use_internal_bytes_contents=use_internal_bytes_contents,
            )
        assert "Inferred mime_type from 'kind' is" in str(exc_info.value)
        with pytest.raises(BadInput) as exc_info:
            await File(mime_type="image/jpeg").from_proto(
                pb.File(kind=123, content=b"asdf"),
                _use_internal_bytes_contents=use_internal_bytes_contents,
            )
        assert "is not a valid File kind." in str(exc_info.value)
        with pytest.raises(BadInput) as exc_info:
            await File(mime_type="image/jpeg").from_proto(
                pb.File(kind=pb.File.FILE_TYPE_JPEG),
                _use_internal_bytes_contents=use_internal_bytes_contents,
            )
        assert "Content is empty!" == str(exc_info.value)


@pytest.mark.asyncio
async def test_exception_to_proto():
    with pytest.raises(BadInput) as exc_info:
        await File(mime_type="multipart/form-data").to_proto(io.BytesIO(b"asdf"))
    assert "which is not allowed to use with 'to_proto'" in str(exc_info.value)
    with pytest.raises(BadInput) as exc_info:
        await File(mime_type="application/bentoml.vnd").to_proto(io.BytesIO(b"asdf"))
    assert "doesn't have a corresponding File 'kind'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_to_proto() -> None:
    assert await File(mime_type="image/bmp").to_proto(io.BytesIO(b"asdf")) == pb.File(
        kind=pb.File.FILE_TYPE_BMP, content=b"asdf"
    )
