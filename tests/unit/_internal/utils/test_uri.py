import os
import typing as t

import psutil
import pytest

WINDOWS_PATHS = [
    r"C:\foo\bar",
    r"C:\foo\bar with space",
    r"C:\\foo\\中文",
    r"relative\path",
    # r"\\localhost\c$\WINDOWS\network",
    # r"\\networkstorage\homes\user",
]
POSIX_PATHS = ["/foo/bar", "/foo/bar with space", "/foo/中文", "relative/path"]


@pytest.fixture()
def example_paths():
    if psutil.WINDOWS:
        return WINDOWS_PATHS
    else:
        return POSIX_PATHS


def test_uri_path_conversion(
    example_paths: t.List[str],  # pylint: disable=redefined-outer-name
) -> None:
    from bentoml._internal.utils.uri import path_to_uri
    from bentoml._internal.utils.uri import uri_to_path

    for path in example_paths:
        restored = uri_to_path(path_to_uri(path))
        assert restored == path or restored == os.path.abspath(path)
