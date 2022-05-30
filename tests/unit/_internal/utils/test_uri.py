from __future__ import annotations

import os
import typing as t
from typing import TYPE_CHECKING
from unittest.mock import patch

import psutil
import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

WINDOWS_PATHS = [
    r"C:\foo\bar",
    r"C:\foo\bar with space",
    r"C:\\foo\\中文",
    r"relative\path",
    # r"\\localhost\c$\WINDOWS\network",
    # r"\\networkstorage\homes\user",
]
POSIX_PATHS = ["/foo/bar", "/foo/bar with space", "/foo/中文", "relative/path"]


@pytest.fixture(scope="function", name="example_paths")
def fixture_example_paths() -> list[str]:
    if psutil.WINDOWS:
        return WINDOWS_PATHS
    else:
        return POSIX_PATHS


@pytest.mark.usefixtures("example_paths")
def test_uri_path_conversion(example_paths: t.List[str]) -> None:
    from bentoml._internal.utils.uri import path_to_uri
    from bentoml._internal.utils.uri import uri_to_path

    for path in example_paths:
        restored = uri_to_path(path_to_uri(path))
        assert restored == path or restored == os.path.abspath(path)


@patch("bentoml._internal.utils.uri.psutil.WINDOWS")
@patch("bentoml._internal.utils.uri.psutil.POSIX")
def test_invalid_os_support(mock_windows: MagicMock, mock_posix: MagicMock) -> None:
    mock_windows.return_value = False
    mock_posix.return_value = False

    from bentoml._internal.utils.uri import path_to_uri

    with pytest.raises(ValueError):
        path_to_uri("/invalid\\/path")
        assert mock_windows.called
        assert mock_posix.called
