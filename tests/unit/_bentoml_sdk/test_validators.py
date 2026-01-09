from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from starlette.datastructures import UploadFile

from _bentoml_sdk.validators import FileSchema


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
