# pylint: disable=redefined-outer-name
from __future__ import annotations

import pytest

from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.bento.build_config import _convert_python_version
from bentoml.exceptions import InvalidArgument


@pytest.mark.parametrize(
    "py_version,expected",
    [
        ("3.9", "3.9"),
        ("3.10", "3.10"),
        ("3.12", "3.12"),
        ("3.8.15", "3.8"),  # micro version is truncated on purpose
    ],
)
def test_convert_python_version_accepts_valid_versions(py_version: str, expected: str):
    assert _convert_python_version(py_version) == expected


def test_convert_python_version_none():
    assert _convert_python_version(None) is None


@pytest.mark.parametrize(
    "py_version",
    ["3.", "3", "", "abc", "3.1a", "3.100", ".9", "3..1", "3.9."],
)
def test_convert_python_version_rejects_invalid_versions(py_version: str):
    with pytest.raises(InvalidArgument):
        _convert_python_version(py_version)


def test_docker_options_rejects_python_version_without_minor():
    # Regression test: "3." used to pass validation and later produced an
    # invalid base image tag such as "python:3.-slim" at build time.
    with pytest.raises(InvalidArgument):
        DockerOptions(python_version="3.")
