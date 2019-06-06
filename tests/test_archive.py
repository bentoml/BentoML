import pytest

from bentoml.archive.archiver import _validate_version_str  # noqa: E402


def test_validate_version_str_fails():
    with pytest.raises(ValueError):
        _validate_version_str('44&')


def test_validate_version_str_pass():
    _validate_version_str('abc_123')
