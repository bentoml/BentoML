from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from starlette.datastructures import Headers
from starlette.datastructures import UploadFile

from _bentoml_sdk.validators import FileSchema
from _bentoml_sdk.validators import PILImageEncoder


@pytest.mark.parametrize(
    "content_type",
    [
        "image/png",
        "image/png; charset=binary",
        "IMAGE/PNG",
        "Image/PNG; name=example.png",
    ],
)
def test_image_upload_content_type_parameters(content_type: str):
    content = io.BytesIO()
    Image.new("RGB", (2, 3), color="red").save(content, format="PNG")
    content.seek(0)
    upload = UploadFile(
        file=content,
        filename="example.png",
        headers=Headers({"content-type": content_type}),
    )

    image = PILImageEncoder().decode(upload)

    assert image.format == "PNG"
    assert image.size == (2, 3)
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize(
    "content_type",
    [
        "image/png",
        "image/png; charset=binary",
        "IMAGE/PNG",
        "Image/PNG; name=example.png",
    ],
)
def test_image_upload_keeps_declared_format_restriction(content_type: str):
    from PIL import UnidentifiedImageError

    content = io.BytesIO()
    Image.new("RGB", (2, 3)).save(content, format="JPEG")
    content.seek(0)
    upload = UploadFile(
        file=content,
        filename="example.png",
        headers=Headers({"content-type": content_type}),
    )

    with pytest.raises(UnidentifiedImageError):
        PILImageEncoder().decode(upload)


def test_file_schema_decode_with_path_separator_in_filename(tmp_path: Path):
    file_content = b"test content"
    upload_file = UploadFile(
        file=io.BytesIO(file_content),
        filename="subdir/nested/document.pdf",
    )

    with patch(
        "bentoml._internal.context.request_temp_dir", return_value=str(tmp_path)
    ):
        result = FileSchema().decode(upload_file)

    assert isinstance(result, Path)
    assert result.exists()
    assert result.read_bytes() == file_content
    assert result.suffix == ".pdf"
